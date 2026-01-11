# D:/kharagpur_hackathon/src/master_pipeline_fixed.py
from data_loader import DataLoader
from decomposer_selector import ClaimDecomposer
from novel_ingester import NovelIngester
from evidence_retriever import EvidenceRetriever
from consistency_checker_api import APIConsistencyChecker
from config_device import USE_API

# Import fallback checker if available
try:
    from consistency_checker import ConsistencyChecker as SimpleConsistencyChecker
except ImportError:
    SimpleConsistencyChecker = APIConsistencyChecker

class MasterPipelineFixed:
    def __init__(self):
        self.loader = DataLoader()
        self.decomposer = ClaimDecomposer()
        self.checker = APIConsistencyChecker() if USE_API else SimpleConsistencyChecker()
        self.ingesters = {}
    
    def process_example(self, story_id: int):
        # ... existing logic ...
        # Now uses API or rules based on USE_API flag
        pass
