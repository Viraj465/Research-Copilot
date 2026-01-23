import os
import re
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from datetime import datetime
from prompts.prompts_template import (SAS_PROCESSOR_SYSTEM_PROMPT, 
                        HAS_PROCESSOR_L1_SYSTEM_PROMPT, 
                        HAS_PROCESSOR_L2_SYSTEM_PROMPT, 
                        HAS_PROCESSOR_L3_SYSTEM_PROMPT, 
                        FINAL_SYNTHESIS_PROMPT)
from utils.llm_factory import LLMFactory
import time

class SectionMetadata(BaseModel):
    """Metadata for a detected section"""
    section_type: str = Field(..., description="Type: abstract, introduction, methodology, results, etc.")
    section_title: str = Field(..., description="Original section title from paper")
    start_position: int = Field(..., description="Character position where section starts")
    word_count: int = Field(..., description="Number of words in this section")
    importance_score: float = Field(..., description="Importance score 0-1")
    contains_figures: bool = Field(default=False, description="Whether section references figures")
    contains_tables: bool = Field(default=False, description="Whether section references tables")
    contains_equations: bool = Field(default=False, description="Whether section has equations")

class SectionSummary(BaseModel):
    """Section-Aware Summary (SAS) output"""
    section_id: str = Field(..., description="Unique section identifier")
    section_type: str = Field(..., description="Section type")
    section_title: str = Field(..., description="Section title")
    
    # Core summary
    executive_summary: str = Field(..., description="1-2 sentence summary")
    detailed_summary: str = Field(..., description="Comprehensive summary")
    
    # Extracted elements
    key_points: List[str] = Field(default=[], description="Key points (3-7 items)")
    methodological_details: List[str] = Field(default=[], description="Methods, algorithms, approaches")
    empirical_findings: List[str] = Field(default=[], description="Results, experiments, metrics")
    technical_terms: List[str] = Field(default=[], description="Important technical terminology")
    citations_mentioned: List[str] = Field(default=[], description="Papers/authors cited")
    
    # Connections
    related_sections: List[str] = Field(default=[], description="References to other sections")
    
    # Quality metrics
    information_density: float = Field(..., description="Information density 0-1")
    novelty_score: float = Field(..., description="Novelty of content 0-1")

class HierarchicalLevel(BaseModel):
    """One level in the hierarchy"""
    level: int = Field(..., description="Hierarchy level (1=lowest, 3=highest)")
    summary: str = Field(..., description="Summary at this abstraction level")
    key_contributions: List[str] = Field(default=[], description="Key contributions at this level")
    # FIX: Made scope optional to prevent validation errors during final synthesis
    scope: Optional[str] = Field(default=None, description="What this level covers")

class ComprehensivePaperAnalysis(BaseModel):
    """Final complete analysis combining SAS + HAS"""
    
    # Paper identification
    paper_title: str = Field(..., description="Paper title")
    authors: List[str] = Field(default=[], description="Author names")
    publication_info: str = Field(default="", description="Publication venue/date")
    
    # Hierarchical summaries (HAS)
    # FIX: Added default=[] to prevent validation errors. Logic manually populates this later.
    hierarchical_summaries: List[HierarchicalLevel] = Field(default=[], description="Multi-level summaries")
    
    # Section summaries (SAS)
    # FIX: Added default={} to prevent validation errors. Logic manually populates this later.
    section_summaries: Dict[str, str] = Field(default={}, description="Summary for each section")
    
    # Comprehensive extraction
    abstract_summary: str = Field(..., description="Abstract summary")
    contributions: List[str] = Field(..., description="Main contributions")
    methodology: Dict[str, Any] = Field(..., description="Methodology details")
    datasets: List[str] = Field(default=[], description="Datasets used")
    experiments: List[str] = Field(default=[], description="Experiments conducted")
    results: Dict[str, Any] = Field(..., description="Key results")
    limitations: List[str] = Field(default=[], description="Limitations")
    future_work: List[str] = Field(default=[], description="Future research directions")
    
    # Technical assessment
    technical_depth: str = Field(..., description="Technical depth assessment")
    novelty: str = Field(..., description="Novelty assessment")
    domain_tags: List[str] = Field(..., description="Research domain tags")
    
    # Resources
    code_resources: Dict[str, Any] = Field(default={}, description="Code/data resources")
    related_papers: List[str] = Field(default=[], description="Related work")
    citations: List[str] = Field(default=[], description="Important citations")
    
    # Metrics
    relevance_score: float = Field(..., description="Overall relevance 0-1")
    quality_score: float = Field(..., description="Paper quality 0-1")
    
    # Metadata
    total_sections: int = Field(..., description="Number of sections analyzed")
    processing_timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


# ============================================================================
# SECTION-AWARE SUMMARIZATION (SAS) ENGINE
# ============================================================================

class SectionAwareSummarizer:
    """
    Section-Aware Summarization (SAS)
    
    Intelligently detects, classifies, and summarizes paper sections
    with awareness of section type and importance.
    """
    
    # Section type patterns and their importance weights
    SECTION_PATTERNS = {
        'abstract': {
            'patterns': [r'\babstract\b', r'\bsummary\b'],
            'importance': 1.0,
            'required_elements': ['problem', 'approach', 'results']
        },
        'introduction': {
            'patterns': [r'\bintroduction\b', r'\b1\.\s*introduction\b'],
            'importance': 0.95,
            'required_elements': ['motivation', 'problem', 'contributions']
        },
        'related_work': {
            'patterns': [r'\brelated work\b', r'\bprior work\b', r'\bliterature review\b'],
            'importance': 0.7,
            'required_elements': ['prior_approaches', 'gaps']
        },
        'background': {
            'patterns': [r'\bbackground\b', r'\bpreliminaries\b'],
            'importance': 0.75,
            'required_elements': ['concepts', 'definitions']
        },
        'methodology': {
            'patterns': [r'\bmethodology\b', r'\bmethod\b', r'\bapproach\b', r'\bmodel\b', r'\barchitecture\b'],
            'importance': 1.0,
            'required_elements': ['approach', 'algorithm', 'implementation']
        },
        'experiments': {
            'patterns': [r'\bexperiments\b', r'\bexperimental setup\b', r'\bevaluation\b'],
            'importance': 0.95,
            'required_elements': ['setup', 'datasets', 'metrics']
        },
        'results': {
            'patterns': [r'\bresults\b', r'\bfindings\b', r'\bperformance\b'],
            'importance': 1.0,
            'required_elements': ['metrics', 'comparisons', 'analysis']
        },
        'discussion': {
            'patterns': [r'\bdiscussion\b', r'\banalysis\b'],
            'importance': 0.85,
            'required_elements': ['interpretation', 'implications']
        },
        'conclusion': {
            'patterns': [r'\bconclusion\b', r'\bconcluding remarks\b'],
            'importance': 0.9,
            'required_elements': ['summary', 'impact', 'future_work']
        },
        'limitations': {
            'patterns': [r'\blimitations\b', r'\bweaknesses\b'],
            'importance': 0.8,
            'required_elements': ['constraints', 'weaknesses']
        },
        'future_work': {
            'patterns': [r'\bfuture work\b', r'\bfuture directions\b'],
            'importance': 0.75,
            'required_elements': ['directions', 'extensions']
        }
    }
    
    def __init__(self, llm):
        self.llm = llm
    
    def detect_sections(self, paper_content: str) -> List[Dict[str, Any]]:
        """
        Detect and classify sections in the paper
        """
        print("Detecting paper sections... 🔍")
        
        sections = []
        lines = paper_content.split('\n')
        current_section = None
        current_content = []
        
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            
            # Check if this is a section header
            detected_type = None
            for section_type, config in self.SECTION_PATTERNS.items():
                for pattern in config['patterns']:
                    if re.search(pattern, line_lower) and len(line.strip()) < 100:
                        detected_type = section_type
                        break
                if detected_type:
                    break
            
            if detected_type:
                # Save previous section
                if current_section and current_content:
                    content_text = '\n'.join(current_content)
                    sections.append({
                        'type': current_section['type'],
                        'title': current_section['title'],
                        'content': content_text,
                        'start_position': current_section['start'],
                        'word_count': len(content_text.split()),
                        'importance': self.SECTION_PATTERNS[current_section['type']]['importance']
                    })
                
                # Start new section
                current_section = {
                    'type': detected_type,
                    'title': line.strip(),
                    'start': i
                }
                current_content = []
            elif current_section:
                current_content.append(line)
        
        # Add last section
        if current_section and current_content:
            content_text = '\n'.join(current_content)
            sections.append({
                'type': current_section['type'],
                'title': current_section['title'],
                'content': content_text,
                'start_position': current_section['start'],
                'word_count': len(content_text.split()),
                'importance': self.SECTION_PATTERNS[current_section['type']]['importance']
            })
        
        print(f"   ✓ SAS: Detected {len(sections)} sections")
        for sec in sections:
            print(f"      - {sec['type']}: {sec['word_count']} words (importance: {sec['importance']})")
        
        return sections
    
    def summarize_section(self, section: Dict[str, Any], max_retries: int = 3) -> SectionSummary:
        """
        Create section-aware summary with type-specific extraction.
        Includes retry logic to handle Groq tool_use_failed errors.
        """
        section_type = section['type']
        section_content = section['content']
        
        # Type-specific prompts
        type_specific_instructions = {
            'abstract': "Extract the problem, approach, key results, and contributions.",
            'introduction': "Extract motivation, problem statement, main contributions, and paper organization.",
            'methodology': "Extract the approach, algorithms, model architecture, and implementation details.",
            'experiments': "Extract experimental setup, datasets, baselines, evaluation metrics, and protocols.",
            'results': "Extract performance metrics, comparisons with baselines, ablation studies, and key findings.",
            'discussion': "Extract interpretation of results, implications, and insights.",
            'conclusion': "Extract main takeaways, impact, and future directions.",
            'related_work': "Extract prior approaches, their limitations, and how this work differs."
        }
        
        instruction = type_specific_instructions.get(
            section_type, 
            "Extract key points, technical details, and important findings."
        )
        
        required_elements_list = self.SECTION_PATTERNS.get(section_type, {}).get('required_elements', [])
        required_elements_str = ', '.join(required_elements_list)

        prompt = ChatPromptTemplate.from_messages([
            ("system", SAS_PROCESSOR_SYSTEM_PROMPT),
            ("user", """Section Title: {title}
                        Section Type: {type}
                        Content:
                        {content}
                        Provide comprehensive section-aware summary.
                    """)])
        
        structured_llm = self.llm.with_structured_output(SectionSummary)
        chain = prompt | structured_llm
        
        # Retry logic to handle Groq tool_use_failed errors
        last_error = None
        for attempt in range(max_retries):
            try:
                # Progressively truncate content on each retry to reduce complexity
                max_chars = 30000 - (attempt * 5000)  # Increased from 8000
                truncated_content = section_content[:max_chars]
                
                summary = chain.invoke({
                    'title': section['title'],
                    'type': section_type,
                    'content': truncated_content,
                    'section_type': section_type.upper(),
                    'required_elements': required_elements_str,
                    'instruction': instruction
                })
                return summary
                
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                # Check for Groq tool_use_failed or similar errors
                if 'tool_use_failed' in error_str or 'failed to call a function' in error_str or '400' in str(e):
                    print(f"         ⚠️ Retry {attempt + 1}/{max_retries}: Groq tool_use error, reducing content size...")
                    time.sleep(2)  # Brief pause before retry
                    continue
                else:
                    # For other errors, raise immediately
                    raise
        
        # If all retries failed, create a fallback minimal summary
        print(f"         ⚠️ All retries failed for {section['title']}, creating fallback summary...")
        return SectionSummary(
            section_id=section_type.upper(),
            section_type=section_type,
            section_title=section['title'],
            executive_summary=f"Summary of {section_type} section (processing error occurred).",
            detailed_summary=f"The {section_type} section could not be fully processed due to API limitations. Original content length: {len(section_content)} chars.",
            key_points=[f"Section type: {section_type}"],
            methodological_details=[],
            empirical_findings=[],
            technical_terms=[],
            citations_mentioned=[],
            related_sections=[],
            information_density=0.5,
            novelty_score=0.5
        )
    
    def _group_sections_globally(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Group ALL sections of the same type together, regardless of position.
        This ensures we only make ~1 API call per section type (e.g. one for all Methodology).
        """
        if not sections:
            return []
        
        # Dictionary to hold merged sections by type
        merged_map = {}
        
        # Order to preserve the first appearance of each type
        type_order = []
        
        for section in sections:
            s_type = section['type']
            
            if s_type not in merged_map:
                merged_map[s_type] = section.copy()
                merged_map[s_type]['title'] = f"{s_type.capitalize()} (Merged)"
                type_order.append(s_type)
            else:
                # Merge content
                merged_map[s_type]['content'] += "\n\n" + section['content']
                merged_map[s_type]['word_count'] += section['word_count']
        
        # Return list in order of first appearance
        return [merged_map[t] for t in type_order]
    
    def process_all_sections(self, sections: List[Dict[str, Any]]) -> List[SectionSummary]:
        """
        Process all sections with section-aware summarization
        """
        print("   📝 SAS: Generating section-aware summaries...")
        
        # Group ALL sections of the same type globally
        grouped_sections = self._group_sections_globally(sections)
        print(f"   📝 SAS: Globally grouped {len(sections)} sections into {len(grouped_sections)} unique types")
        
        summaries = []
        for i, section in enumerate(grouped_sections):
            print(f"      Processing group {i+1}/{len(grouped_sections)}: {section['type']} ({section['word_count']} words)")
            
            # Handle long sections by chunking (reduced threshold to avoid Groq errors)
            if section['word_count'] > 1500:
                print(f"         (Long section with {section['word_count']} words, chunking...)")
                chunk_summaries = self._chunk_and_summarize(section)
                summaries.extend(chunk_summaries)
            else:
                summary = self.summarize_section(section)
                summaries.append(summary)
            
            # Rate limiting sleep
            time.sleep(10)
        
        print(f"   ✓ SAS: Generated {len(summaries)} section summaries")
        return summaries
    
    def _chunk_and_summarize(self, section: Dict[str, Any]) -> List[SectionSummary]:
        """
        Chunk long sections and summarize each chunk
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=10000,
            chunk_overlap=500,
            separators=["\n\n", "\n", ". ", " "]
        )
        
        chunks = splitter.split_text(section['content'])
        chunk_summaries = []
        
        for i, chunk in enumerate(chunks):
            chunk_section = {
                'type': section['type'],
                'title': f"{section['title']} (Part {i+1})",
                'content': chunk,
                'start_position': section['start_position'],
                'word_count': len(chunk.split()),
                'importance': section['importance']
            }
            summary = self.summarize_section(chunk_section)
            chunk_summaries.append(summary)
        
        return chunk_summaries


# ============================================================================
# HIERARCHICAL SUMMARIZATION (HAS) ENGINE
# ============================================================================

class HierarchicalSummarizer:
    """
    Hierarchical Summarization (HAS)
    
    Creates multi-level abstractions:
    - Level 1: Detailed (section-level insights)
    - Level 2: Intermediate (cross-section synthesis)
    - Level 3: Executive (high-level overview)
    """
    
    def __init__(self, llm):
        self.llm = llm
    
    def create_level1_summary(
        self, 
        section_summaries: List[SectionSummary]
    ) -> HierarchicalLevel:
        """
        Level 1: Detailed summary from section summaries
        """
        print("   📊 HAS Level 1: Creating detailed summary...")
        
        # Aggregate all section content
        all_key_points = []
        all_findings = []
        
        for summary in section_summaries:
            all_key_points.extend(summary.key_points)
            all_findings.extend(summary.empirical_findings)
        
        # Create detailed synthesis
        sections_text = "\n\n".join([
            f"**{s.section_type.upper()}**: {s.detailed_summary}"
            for s in section_summaries
        ])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", HAS_PROCESSOR_L1_SYSTEM_PROMPT),
            ("user", """Section Summaries:
                        {sections}
                        All Key Points: {key_points}
                        Create detailed Level 1 summary that preserves technical depth.""")])
        
        response = self.llm.invoke(prompt.format_messages(
            sections=sections_text[:50000], # Increased from 8000
            key_points=str(all_key_points[:100]) # Increased from 50
        ))
        
        level1 = HierarchicalLevel(
            level=1,
            summary=response.content,
            key_contributions=all_key_points[:50], # Increased from 20
            scope="Detailed section-level analysis with technical specifics"
        )
        
        print("   ✓ HAS Level 1 complete")
        return level1
    
    def create_level2_summary(
        self, 
        level1: HierarchicalLevel,
        section_summaries: List[SectionSummary]
    ) -> HierarchicalLevel:
        """
        Level 2: Intermediate synthesis across sections
        """
        print("   📊 HAS Level 2: Creating intermediate synthesis...")
        
        # Group sections by type for cross-section analysis
        methodology_sections = [s for s in section_summaries if s.section_type in ['methodology', 'approach', 'model']]
        results_sections = [s for s in section_summaries if s.section_type in ['results', 'experiments']]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", HAS_PROCESSOR_L2_SYSTEM_PROMPT),
            ("user", """Level 1 (Detailed):
{level1}

Methodology Insights: {methodology}
Results Insights: {results}

Create intermediate Level 2 summary focusing on main contributions and their validation.""")
        ])
        
        methodology_text = " | ".join([s.executive_summary for s in methodology_sections])
        results_text = " | ".join([s.executive_summary for s in results_sections])
        
        response = self.llm.invoke(prompt.format_messages(
            level1=level1.summary[:20000], # Increased from 3000
            methodology=methodology_text[:10000], # Increased from 1000
            results=results_text[:10000] # Increased from 1000
        ))
        
        level2 = HierarchicalLevel(
            level=2,
            summary=response.content,
            key_contributions=level1.key_contributions[:20], # Increased from 10
            scope="Cross-section synthesis of contributions and findings"
        )
        
        print("   ✓ HAS Level 2 complete")
        return level2
    
    def create_level3_summary(
        self, 
        level2: HierarchicalLevel,
        section_summaries: List[SectionSummary]
    ) -> HierarchicalLevel:
        """
        Level 3: Executive summary (highest abstraction)
        """
        print("   📊 HAS Level 3: Creating executive summary...")
        
        # Extract only the most critical information
        abstract_summary = next(
            (s.executive_summary for s in section_summaries if s.section_type == 'abstract'),
            "No abstract found"
        )
        
        contributions = next(
            (s.key_points for s in section_summaries if s.section_type == 'introduction'),
            []
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", HAS_PROCESSOR_L3_SYSTEM_PROMPT),
            ("user", """Abstract: {abstract}
                        Level 2 (Intermediate):
                        {level2}
                        Main Contributions: {contributions}

                        Create concise executive summary answering:
                        1. What problem does this solve?
                        2. What's the proposed solution?
                        3. What are the key results?
                        4. Why does it matter?
                    """)])
        
        response = self.llm.invoke(prompt.format_messages(
            abstract=abstract_summary,
            level2=level2.summary[:20000], # Increased from 2000
            contributions=str(contributions[:20]) # Increased from 5
        ))
        
        level3 = HierarchicalLevel(
            level=3,
            summary=response.content,
            key_contributions=level2.key_contributions[:10], # Increased from 5
            scope="Executive overview for quick understanding"
        )
        
        print("   ✓ HAS Level 3 complete")
        return level3
    
    def create_hierarchy(
        self, 
        section_summaries: List[SectionSummary]
    ) -> List[HierarchicalLevel]:
        """
        Create complete 3-level hierarchy
        """
        print("\n   🏗️  HAS: Building hierarchical summaries...")
        
        level1 = self.create_level1_summary(section_summaries)
        level2 = self.create_level2_summary(level1, section_summaries)
        level3 = self.create_level3_summary(level2, section_summaries)
        
        print("   ✓ HAS: Complete 3-level hierarchy created")
        
        return [level1, level2, level3]


# ============================================================================
# INTEGRATED SAS + HAS PROCESSOR
# ============================================================================

class SASHASProcessor:
    """
    Integrated Section-Aware + Hierarchical Summarization
    
    Complete pipeline:
    1. Section detection and classification
    2. Section-aware summarization (SAS)
    3. Hierarchical synthesis (HAS)
    4. Final comprehensive analysis
    """
    
    def __init__(self, llm_config: Dict = None):
        self.llm = LLMFactory.get_llm(
            agent="paper_analysis",
            temperature=0.1,
            max_retries=5,
            llm_config=llm_config
        )
        
        self.sas = SectionAwareSummarizer(self.llm)
        self.has = HierarchicalSummarizer(self.llm)
    
    def process_paper(self, paper_content: str) -> ComprehensivePaperAnalysis:
        """
        Complete SAS + HAS processing pipeline
        """
        print("\n" + "="*70)
        print("🚀 SAS + HAS PAPER ANALYSIS PIPELINE")
        print("="*70)
        print(f"Paper length: {len(paper_content):,} characters")
        print(f"Estimated words: {len(paper_content.split()):,}")
        
        # ====================================================================
        # PHASE 1: Section-Aware Summarization (SAS)
        # ====================================================================
        print("\n📋 PHASE 1: SECTION-AWARE SUMMARIZATION (SAS)")
        print("-" * 70)
        
        # Detect sections
        sections = self.sas.detect_sections(paper_content)
        
        # Summarize each section
        section_summaries = self.sas.process_all_sections(sections)
        
        # ====================================================================
        # PHASE 2: Hierarchical Summarization (HAS)
        # ====================================================================
        print("\n🏗️  PHASE 2: HIERARCHICAL SUMMARIZATION (HAS)")
        print("-" * 70)
        
        hierarchy = self.has.create_hierarchy(section_summaries)
        
        # ====================================================================
        # PHASE 3: Final Synthesis
        # ====================================================================
        print("\n🔬 PHASE 3: FINAL SYNTHESIS")
        print("-" * 70)
        
        final_analysis = self._synthesize_final_analysis(
            paper_content,
            sections,
            section_summaries,
            hierarchy
        )
        
        print("\n" + "="*70)
        print("✅ SAS + HAS ANALYSIS COMPLETE")
        print("="*70)
        print(f"Total sections analyzed: {len(sections)}")
        print(f"Hierarchical levels: {len(hierarchy)}")
        print(f"Total contributions identified: {len(final_analysis.contributions)}")
        
        return final_analysis
    
    def _synthesize_final_analysis(
        self,
        paper_content: str,
        sections: List[Dict],
        section_summaries: List[SectionSummary],
        hierarchy: List[HierarchicalLevel],
        max_retries: int = 3
    ) -> ComprehensivePaperAnalysis:
        """
        Synthesize everything into final structured output.
        Includes retry logic to handle Groq tool_use_failed errors.
        """
        print("   🔄 Synthesizing final comprehensive analysis...")
        
        # Extract paper metadata
        first_page = paper_content[:2000]
        
        # Aggregate data from sections
        all_contributions = []
        all_methodologies = []
        all_results = []
        all_datasets = []
        all_limitations = []
        all_citations = []
        
        for summary in section_summaries:
            all_contributions.extend(summary.key_points)
            all_methodologies.extend(summary.methodological_details)
            all_results.extend(summary.empirical_findings)
            all_citations.extend(summary.citations_mentioned)
        
        # Build section summaries dict
        section_summaries_dict = {
            s.section_title: s.detailed_summary
            for s in section_summaries
        }
        
        # Create final synthesis prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", FINAL_SYNTHESIS_PROMPT),
            ("user", """First Page (for metadata):
                        {first_page}

                        HIERARCHICAL SUMMARIES:
                        Level 3 (Executive): {level3}
                        Level 2 (Intermediate): {level2}
                        Level 1 (Detailed): {level1}

                        SECTION DATA:
                        Contributions: {contributions}
                        Methodologies: {methodologies}
                        Results: {results}
                        Citations: {citations}
                        Create comprehensive structured analysis.
                    """)])
        
        structured_llm = self.llm.with_structured_output(ComprehensivePaperAnalysis)
        chain = prompt | structured_llm
        
        # Retry logic to handle Groq tool_use_failed errors
        last_error = None
        for attempt in range(max_retries):
            try:
                # Progressively reduce content on each retry
                first_page_len = 10000 - (attempt * 2000)  # Increased from 2000
                level1_len = 20000 - (attempt * 4000)      # Increased from 2000
                contrib_count = 50 - (attempt * 10)        # Increased from 20
                method_count = 30 - (attempt * 5)          # Increased from 15
                result_count = 30 - (attempt * 5)          # Increased from 15
                citation_count = 50 - (attempt * 10)       # Increased from 20
                
                final_analysis = chain.invoke({
                    "first_page": first_page[:first_page_len],
                    "level3": hierarchy[2].summary[:10000], # Increased from 1500
                    "level2": hierarchy[1].summary[:15000], # Increased from 1500
                    "level1": hierarchy[0].summary[:level1_len],
                    "contributions": str(all_contributions[:contrib_count]),
                    "methodologies": str(all_methodologies[:method_count]),
                    "results": str(all_results[:result_count]),
                    "citations": str(all_citations[:citation_count])
                })
                
                # Enrich with hierarchical summaries and section summaries
                # This overwrites any empty/hallucinated lists from the LLM with the valid ones from previous phases
                final_analysis.hierarchical_summaries = hierarchy
                final_analysis.section_summaries = section_summaries_dict
                final_analysis.total_sections = len(sections)
                
                print("   ✓ Final synthesis complete")
                return final_analysis
                
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                # Check for Groq tool_use_failed or similar errors
                if 'tool_use_failed' in error_str or 'failed to call a function' in error_str or '400' in str(e):
                    print(f"   ⚠️ Retry {attempt + 1}/{max_retries}: Groq tool_use error in final synthesis, reducing content...")
                    time.sleep(2)
                    continue
                else:
                    # For other errors, raise immediately
                    raise
        
        # If all retries failed, create a fallback analysis from collected data
        print(f"   ⚠️ All retries failed for final synthesis, creating fallback analysis...")
        
        # Extract title from first page
        title_lines = first_page.split('\n')[:5]
        paper_title = title_lines[0] if title_lines else "Unknown Paper"
        
        fallback_analysis = ComprehensivePaperAnalysis(
            paper_title=paper_title,
            authors=[],
            publication_info="",
            hierarchical_summaries=hierarchy,
            section_summaries=section_summaries_dict,
            abstract_summary=hierarchy[2].summary if len(hierarchy) > 2 else "Analysis completed with partial data.",
            contributions=all_contributions[:20], # Increased from 10
            methodology={"approach": ", ".join(all_methodologies[:10])} if all_methodologies else {}, # Increased from 5
            datasets=[],
            experiments=[],
            results={"findings": ", ".join(all_results[:10])} if all_results else {}, # Increased from 5
            limitations=[],
            future_work=[],
            technical_depth="Moderate (fallback analysis)",
            novelty="See hierarchical summaries for details",
            domain_tags=["Research Paper"],
            code_resources={},
            related_papers=[],
            citations=all_citations[:20], # Increased from 10
            relevance_score=0.7,
            quality_score=0.7,
            total_sections=len(sections)
        )
        
        print("   ✓ Fallback synthesis complete")
        return fallback_analysis