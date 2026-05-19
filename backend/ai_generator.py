import anthropic
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to a comprehensive search tool for course information.

Tool Usage:
- Use **get_course_outline** for questions about a course's structure, lesson list, or overview
- Use **search_course_content** for questions about specific course content or educational material details
- You may make up to 2 sequential tool calls before responding
- Use a second tool call only when the first result reveals you need additional specific information
- Stop calling tools as soon as you have enough information to answer
- Synthesize tool results into accurate, fact-based responses
- If a tool yields no results, state this clearly without offering alternatives

For outline responses, include: course title, course link, and the number and title of each lesson.

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without tools
- **Course-specific questions**: Use the appropriate tool first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        messages = [{"role": "user", "content": query}]
        api_params = {**self.base_params, "messages": messages, "system": system_content}
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}

        response = self.client.messages.create(**api_params)

        MAX_TOOL_ROUNDS = 2
        rounds_remaining = MAX_TOOL_ROUNDS

        while rounds_remaining > 0:
            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
            if not tool_use_blocks or not tool_manager:
                break

            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for block in tool_use_blocks:
                result = tool_manager.execute_tool(block.name, **block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })

            messages.append({"role": "user", "content": tool_results})

            rounds_remaining -= 1
            # api_params["messages"] is the same list object as messages — already updated
            response = self.client.messages.create(**api_params)

        if not response.content:
            return ""
        for block in response.content:
            if block.type == "text":
                return block.text
        return ""