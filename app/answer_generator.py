"""
Answer generation module for the Study Assistant RAG system.

This module provides functionality to generate study-focused answers using
retrieved context from documents, with support for both OpenAI API and local LLMs.
"""

import os
import re
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

from .models import StudyResponse, ResponseStyle

logger = logging.getLogger(__name__)

class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    LOCAL = "local"  # For future Ollama integration

@dataclass
class AnswerConfig:
    """Configuration for answer generation."""
    provider: LLMProvider = LLMProvider.OPENAI
    model_name: str = "gpt-3.5-turbo"
    max_tokens: int = 500
    temperature: float = 0.3
    api_key: Optional[str] = None

class AnswerGenerator:
    """
    Generates study-focused answers using retrieved context and LLM.
    
    Supports multiple LLM providers and response styles optimized for studying.
    """
    
    def __init__(self, config: Optional[AnswerConfig] = None):
        """
        Initialize the answer generator.
        
        Args:
            config: Configuration for LLM provider and parameters
        """
        self.config = config or AnswerConfig()
        self._client = None
        
        # Initialize the appropriate client
        if self.config.provider == LLMProvider.OPENAI:
            self._init_openai_client()
    
    def _init_openai_client(self):
        """Initialize OpenAI client."""
        try:
            import openai
            
            # Get API key from config or environment
            api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.warning("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
                return
            
            self._client = openai.OpenAI(api_key=api_key)
            logger.info("OpenAI client initialized successfully")
            
        except ImportError:
            logger.error("OpenAI library not installed. Run: pip install openai")
            raise RuntimeError("OpenAI library not available")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {str(e)}")
            raise RuntimeError(f"OpenAI initialization failed: {str(e)}")
    
    def _create_study_prompt(self, question: str, context_chunks: List[str], 
                           response_style: ResponseStyle) -> str:
        """
        Create a study-focused prompt for the LLM.
        
        Args:
            question: The student's question
            context_chunks: Retrieved text chunks with source information
            response_style: Desired response style
            
        Returns:
            Formatted prompt string
        """
        # Style-specific instructions
        style_instructions = {
            ResponseStyle.BRIEF: "Provide a concise, direct answer with key points in bullet format.",
            ResponseStyle.DETAILED: "Provide a comprehensive explanation with examples, definitions, and clear structure.",
            ResponseStyle.COMPREHENSIVE: "Provide an in-depth analysis with multiple perspectives, examples, definitions, and detailed explanations."
        }
        
        style_instruction = style_instructions.get(response_style, style_instructions[ResponseStyle.DETAILED])
        
        prompt = f"""You are a helpful study assistant for students. Your role is to answer questions based on the provided study materials.

FORMATTING REQUIREMENTS:
- {style_instruction}
- Use **bold** for key concepts and important terms
- Use bullet points (•) for lists and key points
- Clearly mark definitions with "Definition:" when explaining terms
- Include examples when available, marked with "Example:"
- Structure your response with clear sections when appropriate
- Use numbered lists for step-by-step processes
- Highlight formulas or equations with proper formatting

CONTENT GUIDELINES:
- Focus on educational content that helps with learning and exam preparation
- Extract and highlight key concepts, definitions, and important facts
- Use clear, student-friendly language appropriate for studying
- Include specific examples from the source material when available
- If the context doesn't contain enough information, say so clearly
- Reference page numbers when mentioning specific information

QUESTION: {question}

STUDY MATERIALS:
{chr(10).join(context_chunks)}

Please provide a well-formatted, study-focused answer based on the materials above:"""
        
        return prompt
    
    def _format_study_answer(self, answer: str, sources: List[str]) -> str:
        """
        Format the answer with enhanced study-friendly formatting.
        
        Args:
            answer: Raw answer from LLM
            sources: List of source references
            
        Returns:
            Formatted answer with enhanced study features
        """
        formatted_answer = answer
        
        # Enhance definition formatting
        formatted_answer = re.sub(
            r'Definition:\s*([^.]+\.)',
            r'**📖 Definition:** *\1*',
            formatted_answer
        )
        
        # Enhance example formatting
        formatted_answer = re.sub(
            r'Example:\s*([^.]+\.)',
            r'**💡 Example:** \1',
            formatted_answer
        )
        
        # Format key concepts (terms in bold)
        formatted_answer = re.sub(
            r'\*\*([^*]+)\*\*',
            r'**🔑 \1**',
            formatted_answer
        )
        
        # Format bullet points with better symbols
        formatted_answer = re.sub(r'^[-•]\s+', '• ', formatted_answer, flags=re.MULTILINE)
        
        # Add page references inline when mentioned
        for i, source in enumerate(sources, 1):
            if f"page {i}" not in formatted_answer.lower():
                # Try to add page reference context
                page_match = re.search(r'page (\d+)', source)
                if page_match:
                    page_num = page_match.group(1)
                    # Add page reference to relevant sections
                    formatted_answer = re.sub(
                        r'(According to|Based on|From) (the|your) (document|material|text)',
                        f'\\1 \\2 \\3 (page {page_num})',
                        formatted_answer,
                        count=1
                    )
        
        return formatted_answer
    
    def _extract_key_concepts(self, answer: str) -> List[str]:
        """
        Extract key concepts from the generated answer.
        
        Args:
            answer: Generated answer text
            
        Returns:
            List of key concepts/terms
        """
        concepts = []
        
        # Find terms marked with bold formatting
        bold_terms = re.findall(r'\*\*🔑\s+([^*]+)\*\*', answer)
        concepts.extend(bold_terms)
        
        # Find definition terms
        definition_terms = re.findall(r'\*\*📖 Definition:\*\*[^*]*?\*([^*]+)\*', answer)
        concepts.extend(definition_terms)
        
        # Find quoted terms
        quoted_terms = re.findall(r'"([^"]+)"', answer)
        concepts.extend([term for term in quoted_terms if len(term) > 2])
        
        # Find terms that appear to be definitions (followed by "is" or "are")
        definition_patterns = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:is|are)\s+(?:a|an|the)?\s*([^.]+)', answer)
        for term, definition in definition_patterns:
            if len(term) > 3 and term not in ['The', 'This', 'That', 'These', 'Those']:
                concepts.append(term)
        
        # Find capitalized terms (potential proper nouns/concepts)
        capitalized_terms = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', answer)
        concept_candidates = [term for term in capitalized_terms 
                            if len(term) > 3 and term not in ['The', 'This', 'That', 'These', 'Those', 'Definition', 'Example']]
        concepts.extend(concept_candidates[:5])
        
        # Remove duplicates and return top concepts
        unique_concepts = []
        seen = set()
        for concept in concepts:
            concept_clean = concept.strip()
            if concept_clean and concept_clean.lower() not in seen and len(concept_clean) > 2:
                unique_concepts.append(concept_clean)
                seen.add(concept_clean.lower())
        
        return unique_concepts[:10]
    
    def generate_answer(self, question: str, context_chunks: List[str], 
                       sources: List[str], response_style: ResponseStyle = ResponseStyle.DETAILED) -> StudyResponse:
        """
        Generate a study-focused answer using retrieved context.
        
        Args:
            question: The student's question
            context_chunks: List of retrieved text chunks
            sources: List of source references
            response_style: Desired response style
            
        Returns:
            StudyResponse with generated answer and metadata
            
        Raises:
            RuntimeError: If answer generation fails
        """
        import time
        start_time = time.time()
        
        try:
            if not context_chunks:
                return StudyResponse(
                    answer="I don't have enough information in your uploaded documents to answer this question.",
                    sources=[],
                    processing_time=time.time() - start_time,
                    key_concepts=[]
                )
            
            # Check if OpenAI client is available
            if not self._client:
                # Fallback to simple context-based response
                answer = self._generate_fallback_answer(question, context_chunks)
                key_concepts = self._extract_key_concepts(answer)
                
                return StudyResponse(
                    answer=answer,
                    sources=sources,
                    processing_time=time.time() - start_time,
                    key_concepts=key_concepts
                )
            
            # Create study-focused prompt
            prompt = self._create_study_prompt(question, context_chunks, response_style)
            
            # Generate answer using OpenAI
            response = self._client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful study assistant for students."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            
            raw_answer = response.choices[0].message.content.strip()
            
            # Format the answer with study-friendly enhancements
            formatted_answer = self._format_study_answer(raw_answer, sources)
            
            # Extract key concepts from the formatted answer
            key_concepts = self._extract_key_concepts(formatted_answer)
            
            processing_time = time.time() - start_time
            
            return StudyResponse(
                answer=formatted_answer,
                sources=sources,
                processing_time=processing_time,
                key_concepts=key_concepts
            )
            
        except Exception as e:
            logger.error(f"Failed to generate answer: {str(e)}")
            
            # Fallback to simple response
            fallback_answer = self._generate_fallback_answer(question, context_chunks)
            
            return StudyResponse(
                answer=fallback_answer,
                sources=sources,
                processing_time=time.time() - start_time,
                key_concepts=[]
            )
    
    def _generate_fallback_answer(self, question: str, context_chunks: List[str]) -> str:
        """
        Generate a simple fallback answer when LLM is not available.
        
        Args:
            question: The student's question
            context_chunks: Retrieved text chunks
            
        Returns:
            Simple formatted answer with study-friendly formatting
        """
        answer = f"**📚 Information from your study materials:**\n\n"
        
        for i, chunk in enumerate(context_chunks[:3], 1):
            # Remove source information from chunk for cleaner display
            clean_chunk = re.sub(r'\[Source:.*?\]\n', '', chunk).strip()
            
            # Add basic formatting to the chunk
            formatted_chunk = self._add_basic_formatting(clean_chunk)
            
            answer += f"**{i}.** {formatted_chunk}\n\n"
        
        answer += "---\n\n"
        answer += "**💡 Note:** This is a basic response showing relevant content from your documents. "
        answer += "For enhanced AI-generated answers with better analysis and formatting, please configure an OpenAI API key."
        
        return answer
    
    def _add_basic_formatting(self, text: str) -> str:
        """
        Add basic study-friendly formatting to text.
        
        Args:
            text: Raw text to format
            
        Returns:
            Text with basic formatting applied
        """
        # Highlight potential definitions
        text = re.sub(
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(is|are)\s+(a|an|the)?\s*([^.]+\.)',
            r'**🔑 \1** \2 \3\4',
            text
        )
        
        # Format numbered lists
        text = re.sub(r'^\s*(\d+)\.\s+', r'• **\1.** ', text, flags=re.MULTILINE)
        
        # Format bullet points
        text = re.sub(r'^\s*[-•]\s+', r'• ', text, flags=re.MULTILINE)
        
        return text
    
    def is_available(self) -> bool:
        """
        Check if the answer generator is properly configured and available.
        
        Returns:
            True if generator can produce enhanced answers, False for fallback only
        """
        return self._client is not None
    
    def get_config_info(self) -> Dict[str, Any]:
        """
        Get information about the current configuration.
        
        Returns:
            Dictionary with configuration details
        """
        return {
            "provider": self.config.provider.value,
            "model_name": self.config.model_name,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "available": self.is_available(),
            "fallback_mode": not self.is_available()
        }