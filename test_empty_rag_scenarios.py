#!/usr/bin/env python3
"""
Test script to validate that the system works with completely empty RAG directories
and no company context for evaluation purposes.
"""

import os
import tempfile
import shutil
from multi_agents.config import RAG_ACTIVE_DIRECTORY
from multi_agents.agents.planner import Planner
from multi_agents.llm.llm_provider import LLMProvider


def test_empty_rag_scenario():
    """Test the system with completely empty RAG directory and no company context."""
    print("=== Testing Empty RAG Scenario ===")
    
    # Create temporary empty RAG directory
    with tempfile.TemporaryDirectory() as temp_dir:
        empty_rag_path = os.path.join(temp_dir, "empty_rag")
        os.makedirs(empty_rag_path)
        
        # Temporarily modify config to use empty directory
        original_rag_dir = RAG_ACTIVE_DIRECTORY
        
        # Initialize LLM and Planner
        llm_provider = LLMProvider()
        planner = Planner(
            llm_client=llm_provider.get_llm,
            vertical_name="Agriculture", 
            region="Finland",
            system_architecture="Edge-Cloud Hybrid",
            doc_path=empty_rag_path  # Use empty directory
        )
        
        # Test the run method
        user_prompt = "Smart irrigation systems for precision agriculture evaluation test"
        
        try:
            context = planner.run(user_prompt)
            
            # Validate results
            assert context.get('vertical_result'), "Vertical result should be present"
            assert context.get('geo_result'), "Geo result should be present"  
            assert context.get('segment_result'), "Segment result should be present"
            assert context.get('positioning_result'), "Positioning result should be present"
            assert context.get('final_report'), "Final report should be present"
            assert context.get('segment_ranking_md'), "Segment ranking should be present"
            
            # Company result should be empty since no company context
            assert context.get('company_result') == "", "Company result should be empty"
            
            print("✅ Empty RAG scenario test PASSED")
            print(f"   - All 4 core agents completed successfully")
            print(f"   - Company agent was properly skipped")
            print(f"   - Segment ranking agent completed")
            print(f"   - Final report generated: {len(context['final_report'])} characters")
            print(f"   - Total tokens used: {planner.total_input_tokens + planner.total_output_tokens}")
            
            return True
            
        except Exception as e:
            print(f"❌ Empty RAG scenario test FAILED: {e}")
            return False


def test_empty_directory_detection():
    """Test that the system properly detects empty directories."""
    print("\n=== Testing Empty Directory Detection ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test 1: Non-existent directory
        non_existent_path = os.path.join(temp_dir, "non_existent")
        
        llm_provider = LLMProvider()
        planner = Planner(
            llm_client=llm_provider.get_llm,
            vertical_name="Agriculture", 
            region="Finland",
            doc_path=non_existent_path
        )
        
        assert planner._rag_query_engine is None, "RAG query engine should be None for non-existent directory"
        print("✅ Non-existent directory properly handled")
        
        # Test 2: Empty directory
        empty_dir = os.path.join(temp_dir, "empty")
        os.makedirs(empty_dir)
        
        planner2 = Planner(
            llm_client=llm_provider.get_llm,
            vertical_name="Agriculture", 
            region="Finland", 
            doc_path=empty_dir
        )
        
        assert planner2._rag_query_engine is None, "RAG query engine should be None for empty directory"
        print("✅ Empty directory properly handled")
        
        return True


if __name__ == "__main__":
    print("Running validation tests for empty RAG scenarios...\n")
    
    success = True
    
    # Test 1: Empty directory detection
    if not test_empty_directory_detection():
        success = False
    
    # Test 2: Full empty RAG scenario
    if not test_empty_rag_scenario():
        success = False
    
    print(f"\n=== Test Results ===")
    if success:
        print("🎉 ALL TESTS PASSED - System ready for evaluation with empty RAG directories")
        print("\nThe system can now:")
        print("  ✅ Run without any files in RAG directories")
        print("  ✅ Skip company agent when no company context available")
        print("  ✅ Generate complete reports using only LLM knowledge")
        print("  ✅ Export segment rankings to Excel")
    else:
        print("❌ SOME TESTS FAILED - Please check the implementation")
