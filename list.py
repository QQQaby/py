"""
生产级测试用例自动维护 Agent 系统
融合长链推理、多 Agent 协作、RAG、人工审核闭环
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple

# ---------- 第三方库（真实环境安装）----------
# pip install openai chromadb gitpython
# --------------------------------------------------
import openai
from chromadb import Client as ChromaClient
from chromadb.utils import embedding_functions

# ---------- 配置与日志 ----------
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(name)s: %(message)s')
logger = logging.getLogger("TestMaintenanceAgent")

# 从环境变量读取配置，方便生产部署
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = "gpt-4o"
VECTOR_DB_PATH = "./test_case_vectors"
TOKEN_CONSUMPTION = {"total": 0}  # 统计 token 消耗

# ---------- 数据模型 ----------
@dataclass
class CodeChange:
    repo: str
    branch: str
    commit_id: str
    changed_files: List[str]
    diff_content: str
    raw_metadata: dict = field(default_factory=dict)

@dataclass
class TestCase:
    id: str
    title: str
    steps: str            # 文本步骤
    expected: str
    module: str
    tags: List[str]
    last_run_status: str  # pass/fail
    embedding: Optional[List[float]] = None  # 向量

@dataclass
class ImpactReport:
    affected_functions: List[str]
    affected_modules: List[str]
    call_chain: List[str]               # 长链推理路径
    risk_score: float
    reasoning: str

@dataclass
class RewriteSuggestion:
    original_id: str
    new_title: str
    new_steps: str
    new_expected: str
    reason: str
    diff: str

# ---------- 基础设施：带重试和计费的LLM调用 ----------
class LLMClient:
    def __init__(self, model=LLM_MODEL):
        self.model = model
        if not OPENAI_API_KEY:
            logger.warning("未设置OPENAI_API_KEY，将使用模拟LLM")
        openai.api_key = OPENAI_API_KEY

    async def chat_complete(self, system_prompt: str, user_prompt: str,
                            temperature=0.2, max_tokens=2000) -> Tuple[str, int]:
        """调用 LLM 并返回 (内容, 消耗token数)"""
        if not OPENAI_API_KEY:
            # 模拟返回，用于测试
            await asyncio.sleep(0.1)
            return self._mock_response(user_prompt), 0
        try:
            resp = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            content = resp.choices[0].message.content
            usage = resp.usage.total_tokens
            TOKEN_CONSUMPTION["total"] += usage
            return content, usage
        except Exception as e:
            logger.error(f"LLM调用失败: {e}")
            raise

    def _mock_response(self, prompt: str) -> str:
        if "impact analysis" in prompt:
            return json.dumps({
                "affected_functions": ["login()", "validate_session()"],
                "call_chain": ["login() change -> session token format -> auth middleware -> /api/user"],
                "risk_score": 0.8
            })
        if "retrieve similar" in prompt:
            return json.dumps({"relevant_ids": ["TC001", "TC003", "TC005"]})
        if "rewrite" in prompt:
            return json.dumps({
                "new_title": "Login with OAuth2",
                "new_steps": "1. Open login page\n2. Click 'Login with Google'\n3. Authorize app",
                "new_expected": "Session token stored, redirect to dashboard"
            })
        return "{}"

# ---------- Agent 基类 ----------
class Agent:
    def __init__(self, name: str, llm: LLMClient):
        self.name = name
        self.llm = llm
        self.logger = logging.getLogger(name)

    def log(self, msg):
        self.logger.info(msg)

# ---------- 具体 Agent ----------

class CodeAnalyzerAgent(Agent):
    """代码影响分析 Agent（长链推理核心）"""
    async def analyze(self, change: CodeChange) -> ImpactReport:
        self.log("正在构建调用图并推理影响链...")
        system = "你是一位资深的代码架构师。根据Git diff，分析影响的函数、模块，并推理出可能受影响的上下游调用链。返回JSON。"
        prompt = f"""
        Repository: {change.repo}
        Branch: {change.branch}
        Changed files: {change.changed_files}
        Diff:
        {change.diff_content[:3000]}

        Provide a detailed impact analysis in JSON with fields:
        - affected_functions: list of function names changed
        - affected_modules: list of modules
        - call_chain: list of strings describing the chain from low-level change to high-level API impact
        - risk_score: 0-1 float
        - reasoning: brief explanation
        """
        content, tokens = await self.llm.chat_complete(system, prompt)
        data = json.loads(content)
        report = ImpactReport(
            affected_functions=data.get("affected_functions", []),
            affected_modules=data.get("affected_modules", []),
            call_chain=data.get("call_chain", []),
            risk_score=data.get("risk_score", 0.5),
            reasoning=data.get("reasoning", "")
        )
        self.log(f"影响链: {' -> '.join(report.call_chain)}")
        return report

class TestRetrieverAgent(Agent):
    """基于向量相似度的相关测试用例检索"""
    def __init__(self, name, llm, vector_db):
        super().__init__(name, llm)
        self.vector_db = vector_db

    async def retrieve(self, impact: ImpactReport, top_k=5) -> List[TestCase]:
        self.log("在向量数据库中检索语义相似的测试用例...")
        # 将影响分析拼接为查询文本
        query_text = f"{impact.affected_functions} {impact.affected_modules} {' '.join(impact.call_chain)}"
        # 此处可使用嵌入模型转换，简化使用关键字或模拟
        if not OPENAI_API_KEY:
            # 模拟：返回几个带有这些标签的用例
            return self._mock_retrieve(impact)
        # 真正检索
        results = self.vector_db.query(query_texts=[query_text], n_results=top_k)
        ids = results['ids'][0] if results['ids'] else []
        # 从元数据重建 TestCase（实际中已存储）
        # 简化为返回存储的对象
        found = []
        for case_id in ids:
            # 模拟构建
            found.append(TestCase(case_id, f"Test for {case_id}", "", "", "", [], "pass"))
        return found

    def _mock_retrieve(self, impact) -> List[TestCase]:
        # 模拟从用例库中查找
        return [
            TestCase("TC001", "Login with email", "1. Open login\n2. Enter creds", "Login ok", "auth", ["login"], "pass"),
            TestCase("TC003", "API user profile", "1. Authenticate\n2. GET /api/user", "200 OK", "api", ["/api/user"], "pass"),
        ]

class TestRewriterAgent(Agent):
    """基于影响链和旧用例生成新用例"""
    async def rewrite(self, case: TestCase, change: CodeChange, impact: ImpactReport) -> RewriteSuggestion:
        self.log(f"正在改写测试用例 {case.id} ...")
        system = "你是一位测试专家。根据代码变更和影响链，改写或生成一个新的测试用例。输出严格JSON。"
        prompt = f"""
        Original Test Case:
        Title: {case.title}
        Steps: {case.steps}
        Expected: {case.expected}

        Code Change:
        Diff: {change.diff_content[:2000]}

        Impact Chain: {' -> '.join(impact.call_chain)}

        Generate a new test case that covers the changes. Output JSON with:
        - new_title
        - new_steps (multi-line string)
        - new_expected
        - reason for change
        """
        content, tokens = await self.llm.chat_complete(system, prompt)
        data = json.loads(content)
        suggestion = RewriteSuggestion(
            original_id=case.id,
            new_title=data.get("new_title", case.title),
            new_steps=data.get("new_steps", case.steps),
            new_expected=data.get("new_expected", case.expected),
            reason=data.get("reason", ""),
            diff=json.dumps(data, indent=2)
        )
        return suggestion

class TestExecutorAgent(Agent):
    """在沙箱中执行测试并返回结果"""
    async def execute(self, case: RewriteSuggestion) -> bool:
        self.log(f"执行测试: {case.new_title}")
        # 实际会调用 pytest/selenium 等，这里模拟执行
        # 根据改写原因中的关键词决定通过/失败，模拟真实情况
        if "OAuth" in case.reason or "new field" in case.reason:
            success = True
        else:
            success = False
        self.log(f"结果: {'PASS' if success else 'FAIL'}")
        return success

class PRGeneratorAgent(Agent):
    """根据改写结果和测试结果生成 Pull Request"""
    async def create_pr(self, suggestions: List[RewriteSuggestion],
                        results: Dict[str, bool]) -> str:
        self.log("生成测试用例更新PR...")
        passed = [forin suggestions if results.get(s.original_id, False)]
        failed = [forin suggestions if not results.get(s.original_id, False)]
        pr_body = f"## 自动生成的测试用例更新\n"
        pr_body += f"### 变更来源\nCommit: {os.getenv('COMMIT_SHA', 'unknown')}\n"
        pr_body += f"### ✅ 通过的用例 ({len(passed)}个)\n"
        forin passed:
            pr_body += f"- {s.new_title}\n  ""steps: {s.new_steps}\n"
        pr_body += f"### ❌ 需要人工审核的用例 ({len(failed)}个)\n"
        forin failed:
            pr_body += f"- **{s.new_title}** (原ID:{s.original_id})\n  ""原因: {s.reason}\n"
        pr_body += "\n> 自动生成，如有问题请手动修正。"
        # 实际调用Git平台API（GitHub/GitLab）创建PR
        # requests.post(...)
        self.log("PR已创建。")
        return pr_body

class HumanReviewAgent(Agent):
    """人工审核节点：失败的用例自动分配给QA团队"""
    async def request_review(self, failed_cases: List[RewriteSuggestion]):
        if not failed_cases:
            return
        self.log(f"通知人工审核 {len(failed_cases)} 个用例...")
        # 发送消息到Slack/飞书/Jira
        # 例如: send_slack(f"请审核以下测试用例: {[c.original_id for c in failed_cases]}")
        pass

# ---------- 编排器 ----------
class TestMaintenanceOrchestrator:
    def __init__(self, llm: LLMClient):
        self.llm = llm
        # 初始化向量数据库（内存模式，生产用持久化）
        self.vector_db = ChromaClient()
        self.collection = self.vector_db.get_or_create_collection("test_cases")
        # Agent实例化
        self.analyzer = CodeAnalyzerAgent("CodeAnalyzer", llm)
        self.retriever = TestRetrieverAgent("TestRetriever", llm, self.collection)
        self.rewriter = TestRewriterAgent("TestRewriter", llm)
        self.executor = TestExecutorAgent("TestExecutor", llm)
        self.pr_generator = PRGeneratorAgent("PRGenerator", llm)
        self.human_review = HumanReviewAgent("HumanReview", llm)

        self.metrics = {
            "cases_affected": 0,
            "cases_rewritten": 0,
            "cases_passed": 0,
            "cases_failed": 0,
            "total_tokens": 0,
            "pipeline_start": None
        }

    async def run(self, change: CodeChange):
        self.metrics["pipeline_start"] = datetime.now()
        logger.info("========== Test Maintenance Pipeline 启动 ==========")

        # 1. 影响分析
        impact = await self.analyzer.analyze(change)
        self.metrics["impact_chain_length"] = len(impact.call_chain)

        # 2. 检索相关用例
        related = await self.retriever.retrieve(impact)
        self.metrics["cases_affected"] = len(related)
        if not related:
            logger.info("无相关测试用例，流水线结束。")
            return

        # 3. 改写用例
        suggestions = []
        for case in related:
            sug = await self.rewriter.rewrite(case, change, impact)
            suggestions.append(sug)
        self.metrics["cases_rewritten"] = len(suggestions)

        # 4. 执行测试
        results = {}
        for sug in suggestions:
            success = await self.executor.execute(sug)
            results[sug.original_id] = success
            if success:
                self.metrics["cases_passed"] += 1
            else:
                self.metrics["cases_failed"] += 1

        # 5. 生成PR
        pr_url = await self.pr_generator.create_pr(suggestions, results)
        self.metrics["pr_url"] = pr_url

        # 6. 人工审核回路：失败的测试通知QA
        failed_suggestions = [forin suggestions if not results[s.original_id]]
        await self.human_review.request_review(failed_suggestions)

        # 7. 收集 Token 消耗
        self.metrics["total_tokens"] = TOKEN_CONSUMPTION["total"]

        # 8. 输出报告
        self._print_report()

    def _print_report(self):
        duration = (datetime.now() - self.metrics["pipeline_start"]).total_seconds()
        print("\n========== Pipeline 完成报告 ==========")
        print(f"耗时: {duration:.2f} 秒")
        print(f"影响链深度: {self.metrics.get('impact_chain_length', 0)}")
        print(f"相关用例数: {self.metrics['cases_affected']}")
        print(f"改写用例数: {self.metrics['cases_rewritten']}")
        print(f"通过: {self.metrics['cases_passed']}, 失败: {self.metrics['cases_failed']}")
        print(f"总 Token 消耗: {self.metrics['total_tokens']}")
        print(f"PR 地址: {self.metrics.get('pr_url', 'N/A')}")
        print("==========================================\n")

# ---------- 示例入口 ----------
async def main():
    # 模拟代码变更
    change = CodeChange(
        repo="backend-api",
        branch="feature/oauth2-login",
        commit_id="ab12cd34",
        changed_files=["auth/handler.py", "auth/models.py"],
        diff_content="""@@ -12,7 +12,8 @@ def login(request):
    -    username = request.POST['username']
    -    password = request.POST['password']
    +    token = request.POST.get('oauth_token')
    +    user = authenticate_oauth(token)
    +    session_id = create_session(user, method='oauth')
    """,
    )

    # 初始化 LLM 客户端
    llm = LLMClient()
    # 运行编排器
    orchestrator = TestMaintenanceOrchestrator(llm)
    await orchestrator.run(change)

if __name__ == "__main__":
    asyncio.run(main())
