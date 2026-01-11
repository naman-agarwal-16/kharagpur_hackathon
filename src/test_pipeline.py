"""
Test the full pipeline: Claim Decomposition -> Novel Ingestion -> Evidence Search
"""

from claim_decomposer import ClaimDecomposer
from novel_ingester import NovelIngester

def test_full_pipeline():
    """Test the complete workflow"""
    
    print("="*80)
    print("FULL PIPELINE TEST")
    print("="*80)
    
    # Step 1: Decompose backstory into claims
    print("\n[STEP 1] Decomposing backstory into claims...")
    print("-"*80)
    
    sample_backstory = """
    Sarah grew up in a strict household where her father's harsh criticism left her with deep self-doubt. 
    She learned to fear abandonment early on, and now avoids close relationships. Despite this, she's 
    incredibly brave when protecting others, though she doesn't see it in herself. She became an orphan 
    at age twelve when her parents died in a car crash, which she rarely talks about.
    """
    
    decomposer = ClaimDecomposer()
    claims = decomposer.decompose(sample_backstory, "Sarah")
    
    print(f"\n✓ Extracted {len(claims)} claims")
    for i, claim in enumerate(claims[:3], 1):  # Show first 3
        print(f"\n  Claim {i}: {claim['claim_id']}")
        print(f"  Text: {claim['claim_text']}")
        print(f"  Type: {claim['claim_type']}")
        print(f"  Confidence: {claim['confidence']:.2f}")
        print(f"  Search terms: {claim['search_vocabulary'][:5]}")
    
    # Step 2: Ingest novel
    print("\n\n[STEP 2] Ingesting novel...")
    print("-"*80)
    
    novel_path = "D:/kharagpur_hackathon/data/sample_novel.txt"
    ingester = NovelIngester(novel_path)
    chunks = ingester.ingest(chunk_method="chapter")
    
    print(f"\n✓ Novel ingested: {len(chunks)} chunks")
    print(f"✓ Characters found: {list(ingester.character_positions.keys())}")
    
    # Step 3: Search for evidence for each claim
    print("\n\n[STEP 3] Searching for evidence...")
    print("-"*80)
    
    if claims:
        for idx, claim in enumerate(claims[:3], 1):  # Test first 3 claims
            print(f"\n\n>>> CLAIM {idx}: {claim['claim_id']}")
            print(f"    {claim['claim_text']}")
            print(f"    Searching with vocabulary: {claim['search_vocabulary'][:3]}...")
            
            # Create vocab dict for search
            vocab_dict = {
                'positive': claim['search_vocabulary'],
                'negative': claim['anti_vocabulary'],
                'patterns': claim['syntactic_patterns']
            }
            
            matches = ingester.search_character("Sarah", vocab_dict)
            
            if matches:
                print(f"\n    ✓ Found {len(matches)} evidence passages")
                
                # Show top 2 matches
                for i, match in enumerate(matches[:2], 1):
                    print(f"\n    Evidence {i} (Score: {match['score']:.1f}):")
                    print(f"    Chunk: {match['chunk_id']} | Scene: {match['scene_type']}")
                    print(f"    Matched terms: {match['matched_terms'][:3]}")
                    print(f"    Text: {match['text'][:200]}...")
            else:
                print(f"    ✗ No evidence found")
    
    print("\n\n" + "="*80)
    print("PIPELINE TEST COMPLETE")
    print("="*80)
    
    return claims, ingester


if __name__ == "__main__":
    claims, ingester = test_full_pipeline()
    
    print("\n\n[SUMMARY]")
    print(f"  - Claims extracted: {len(claims)}")
    print(f"  - Novel chunks: {len(ingester.chunks)}")
    print(f"  - Characters tracked: {len(ingester.character_positions)}")
    print("\n  Pipeline is ready for real data processing!")
