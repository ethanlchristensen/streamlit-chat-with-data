"""
Enhanced pandas dataframe accessor for LLM-powered data analysis.
Based on the original work from https://github.com/rvanasa/pandas-gpt
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import asyncio
from enum import Enum
from typing import Optional, AsyncIterator, Tuple, Any, Dict, List, Union
import streamlit as st
import time
import io
import re
from abc import ABC, abstractmethod
from textwrap import dedent

# Global configuration
DEFAULT_MODEL = "gemma3:12b"
DEFAULT_MUTABLE = False
DEFAULT_VERBOSE = False

# LLM response template
PROMPT_TEMPLATE = """
Write a Python function `process({arg_name})` which takes the following input value:

{arg_name} = {arg}

This is the function's purpose: {goal}
"""

# System prompt for code generation
CODE_SYSTEM_PROMPT = """
Write the function in a Python code block with all necessary imports and no example usage.
You are allowed to create functions that return the following values:
- Plotly Express Figure
- Modified Dataframe
- String
- Series
- Number Calculations
- Or any combination of the above in a list or dictionary

RULES:
- You can only generate code that deals with analyzing the data.
- If the user requests something not related to analyzing the data, generate a function that returns a string kindly declining the request.
- You make all charts using Plotly Express python package.
- Provide the Code and ONLY the code. 
- DO NOT provide a description of the code.
- DO NOT provide a walk through of the code.
- Ensure all generated plotly charts are dark mode.
- If you need to return multiple items, use a dictionary which contains a {"title": "<title for the object>", "data": result}
- If you created a Ploty figure, ALWAYS return the figure. Do not call .show(), .to_json(), or any other functions. Just return the figure.
"""

# System prompt for description generation
DESCRIPTION_SYSTEM_PROMPT = """
Turn this python function into 1-5 bullet points describing the activity you are going to take.
"""

# Cache for LLM responses
_ask_cache = {}


class ResultKind(Enum):
    """Enum for different types of results during the async execution flow."""
    START = "start"
    CODE_BLOCK = "code_block"
    DESCRIPTION = "description"
    RESULT = "result"
    END = "end"


class AskResult:
    """Container for results yielded during async execution."""
    def __init__(self, kind: ResultKind, content: Any):
        self.kind = kind
        self.content = content


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    async def generate(self, prompt: str, system_prompt: str) -> str:
        """Generate a completion from the LLM provider."""
        pass


class OllamaProvider(LLMProvider):
    """Ollama-specific implementation of LLM provider."""
    
    def __init__(self, model: str, host: str = "localhost"):
        self.model = model
        self.host = host
    
    async def generate(self, prompt: str, system_prompt: str) -> str:
        """Generate a completion using Ollama."""
        import ollama
        
        ollama_client = ollama.Client(host=self.host)
        completion = ollama_client.chat(
            model=self.model,
            messages=[
                dict(role="system", content=system_prompt),
                dict(role="user", content=prompt),
            ],
            stream=False
        )
        
        return completion.message.content


class OpenAIProvider(LLMProvider):
    """OpenAI-specific implementation of LLM provider."""
    
    def __init__(self, model: str, api_key: str):
        self.model = model
        self.api_key = api_key
    
    async def generate(self, prompt: str, system_prompt: str) -> str:
        """Generate a completion using OpenAI."""
        from openai import AsyncOpenAI
        
        client = AsyncOpenAI(api_key=self.api_key)
        completion = await client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
        )
        
        return completion.choices[0].message.content


class AnthropicProvider(LLMProvider):
    """Anthropic-specific implementation of LLM provider."""
    
    def __init__(self, model: str, api_key: str):
        self.model = model
        self.api_key = api_key
    
    async def generate(self, prompt: str, system_prompt: str) -> str:
        """Generate a completion using Anthropic."""
        import anthropic
        
        client = anthropic.AsyncAnthropic(api_key=self.api_key)
        message = await client.messages.create(
            model=self.model,
            system=system_prompt,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        return message.content[0].text


class GoogleProvider(LLMProvider):
    """Google Gemini-specific implementation of LLM provider."""
    
    def __init__(self, model: str, api_key: str):
        self.model = model
        self.api_key = api_key
    
    async def generate(self, prompt: str, system_prompt: str) -> str:
        """Generate a completion using Google's Gemini API."""
        import google.generativeai as genai
        
        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(model_name=self.model)
        
        response = await model.generate_content_async(
            [system_prompt, prompt]
        )
        
        return response.text


class LLMProviderFactory:
    """Factory for creating LLM providers."""
    
    @staticmethod
    def create_provider(provider_type: str, model: str, **kwargs) -> LLMProvider:
        """Create and return an LLM provider based on the specified type."""
        if provider_type.lower() == "ollama":
            host = kwargs.get("host", "localhost")
            return OllamaProvider(model=model, host=host)
        
        elif provider_type.lower() == "openai":
            api_key = kwargs.get("api_key")
            if not api_key:
                raise ValueError("API key is required for OpenAI provider")
            return OpenAIProvider(model=model, api_key=api_key)
        
        elif provider_type.lower() == "anthropic":
            api_key = kwargs.get("api_key")
            if not api_key:
                raise ValueError("API key is required for Anthropic provider")
            return AnthropicProvider(model=model, api_key=api_key)
        
        elif provider_type.lower() == "google":
            api_key = kwargs.get("api_key")
            if not api_key:
                raise ValueError("API key is required for Google provider")
            return GoogleProvider(model=model, api_key=api_key)
        
        else:
            raise ValueError(f"Unsupported provider type: {provider_type}")


class Ask:
    """Main class for handling LLM-powered data analysis requests."""
    
    def __init__(
        self, 
        provider_type: str = "ollama",
        model: str = DEFAULT_MODEL, 
        verbose: bool = DEFAULT_VERBOSE, 
        mutable: bool = DEFAULT_MUTABLE,
        **provider_kwargs
    ):
        self.verbose = verbose
        self.mutable = mutable
        self.description = None
        self.function_block = None
        self.provider = LLMProviderFactory.create_provider(
            provider_type=provider_type,
            model=model,
            **provider_kwargs
        )

    @staticmethod
    def _fill_template(template: str, **kwargs) -> str:
        """Fill in a template string with the provided values."""
        result = dedent(template.lstrip("\n").rstrip())
        for k, v in kwargs.items():
            result = result.replace(f"{{{k}}}", v)
        
        # Check for any remaining unfilled template variables
        m = re.match(r"\{[a-zA-Z0-9_]*\}", result)
        if m:
            raise ValueError(f"Expected variable: {m.group(0)}")
        
        return result

    def _get_prompt(self, goal: str, arg: Any) -> str:
        """Construct the prompt for the LLM."""
        if isinstance(arg, (pd.DataFrame, pd.Series)):
            buf = io.StringIO()
            arg.info(buf=buf)
            arg_summary = buf.getvalue()
        else:
            arg_summary = repr(arg)
        
        arg_name = (
            "df" if isinstance(arg, pd.DataFrame) 
            else "index" if isinstance(arg, pd.Index) 
            else "data"
        )

        return self._fill_template(
            PROMPT_TEMPLATE, 
            arg_name=arg_name, 
            arg=arg_summary.strip(), 
            goal=goal.strip()
        )

    async def _run_prompt_async(self, prompt: str, system_prompt: str) -> str:
        """Run the prompt through the LLM provider asynchronously."""
        if prompt in _ask_cache:
            return _ask_cache[prompt]
        
        response = await self.provider.generate(prompt, system_prompt)
        _ask_cache[prompt] = response
        return response

    def _run_prompt(self, prompt: str, system_prompt: str) -> str:
        """Run the prompt through the LLM provider synchronously."""
        if prompt in _ask_cache:
            return _ask_cache[prompt]
        
        # Create event loop for async operation if needed
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        response = loop.run_until_complete(
            self.provider.generate(prompt, system_prompt)
        )
        
        _ask_cache[prompt] = response
        return response

    async def _extract_code_block_async(self, text: str) -> str:
        """Extract the code block from the LLM response."""
        pattern = r"```(\s*(py|python)\s*\n)?([\s\S]*?)```"
        m = re.search(pattern, text)
        if not m:
            return text
        
        code_block = m.group(3)
        self.function_block = code_block
        return code_block

    def _extract_code_block(self, text: str) -> str:
        """Extract the code block from the LLM response synchronously."""
        pattern = r"```(\s*(py|python)\s*\n)?([\s\S]*?)```"
        m = re.search(pattern, text)
        if not m:
            return text
        
        self.description = self._run_prompt(
            m.group(3),
            DESCRIPTION_SYSTEM_PROMPT,
        )
        self.function_block = m.group(3)
        return m.group(3)

    def _eval(self, source: str, *args) -> Tuple[Any, str, str]:
        """Evaluate the generated code with the input arguments."""
        scope = {"_args_": args}
        exec(
            self._fill_template(
                """
                {source}
                _result_ = process(*_args_)
                """,
                source=source,
            ),
            scope,
        )
        return scope["_result_"], self.description, self.function_block


    async def _eval_async(self, source: str, *args) -> Any:
        """Evaluate the generated code with the input arguments asynchronously."""
        scope = {"_args_": args}
        try:
            exec(
                self._fill_template(
                    """
                    {source}
                    _result_ = process(*_args_)
                    """,
                    source=source,
                ),
                scope,
            )
            return scope["_result_"]
        except Exception as e:
            # Capture and format the error
            import traceback
            error_message = f"Error in generated code: {str(e)}\n"
            error_traceback = traceback.format_exc().split('\n')
            # Only include relevant parts of the traceback
            filtered_traceback = [line for line in error_traceback if "<string>" in line]
            if filtered_traceback:
                error_message += "In the generated function:\n" + "\n".join(filtered_traceback)
            
            # Return error as a formatted string that can be displayed to the user
            return {"error": error_message, "title": "Error in Analysis"}

    def _code(self, goal: str, arg: Any) -> str:
        """Generate code to process the input data."""
        prompt = self._get_prompt(goal, arg)
        result = self._run_prompt(prompt, CODE_SYSTEM_PROMPT)
        
        if self.verbose:
            print("\n", result)
            
        return self._extract_code_block(result)

    async def _code_async(self, goal: str, arg: Any) -> str:
        """Generate code to process the input data asynchronously."""
        prompt = self._get_prompt(goal, arg)
        result = await self._run_prompt_async(prompt, CODE_SYSTEM_PROMPT)
        
        if self.verbose:
            print("\n", result)
            
        return await self._extract_code_block_async(result)

    async def _get_description_async(self, code_block: str) -> str:
        """Generate a description of what the code does."""
        description = await self._run_prompt_async(code_block, DESCRIPTION_SYSTEM_PROMPT)
        self.description = description
        return description

    def code(self, goal: str, arg: Any) -> None:
        """Print the generated code for the given goal and input."""
        print(self._code(goal, arg))

    def prompt(self, goal: str, arg: Any) -> None:
        """Print the prompt that would be sent to the LLM."""
        print(self._get_prompt(goal, arg))

    def __call__(self, goal: str, *args) -> Tuple[Any, str, str]:
        """Process the input with the given goal and return the result."""
        source = self._code(goal, *args)
        return self._eval(source, *args)
    
    async def __call_async__(self, goal: str, *args) -> AsyncIterator[AskResult]:
        """Process the input asynchronously, yielding results at each step."""
        # First yield an opening result
        yield AskResult(ResultKind.START, None)

        # Generate and yield the code block
        with st.spinner("⚙️ Generating some code to help"):
            code_block = await self._code_async(goal, *args)
            yield AskResult(ResultKind.CODE_BLOCK, code_block)
        
        # Generate and yield the description
        with st.spinner("💡 Creating a walkthrough for the code"):
            description = await self._get_description_async(code_block)
            time.sleep(0.5)  # Small delay for UI feedback
            yield AskResult(ResultKind.DESCRIPTION, description)
        
        # Evaluate the code and yield the result
        with st.spinner("🛠️ Executing the code to get you results"):
            result = await self._eval_async(code_block, *args)
            time.sleep(0.5)  # Small delay for UI feedback
            yield AskResult(ResultKind.RESULT, result)

        # Finally yield an end result
        yield AskResult(ResultKind.END, None)


@pd.api.extensions.register_dataframe_accessor("ask")
@pd.api.extensions.register_series_accessor("ask")
@pd.api.extensions.register_index_accessor("ask")
class AskAccessor:
    """Pandas accessor that adds LLM-powered analysis capabilities to dataframes, series, and indices."""
    
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def _ask(self, **kwargs) -> Ask:
        """Create a new Ask instance with the given parameters."""
        return Ask(**kwargs)

    def _data(self, **kwargs) -> pd.DataFrame:
        """Return a copy of the data if not mutable."""
        if not DEFAULT_MUTABLE and not kwargs.get("mutable") and hasattr(self._obj, "copy"):
            return self._obj.copy()
        return self._obj

    def __call__(self, goal: str, *args, **kwargs) -> Tuple[Any, str, str]:
        """Process the pandas object with the given goal and return the result."""
        ask = self._ask(**kwargs)
        data = self._data(**kwargs)
        return ask(goal, data, *args)
    
    # Instead of using __aiter__ which makes the object itself an async iterator
    async def __call_async__(self, goal: str, *args, **kwargs) -> AsyncIterator[AskResult]:
        """Process the pandas object asynchronously, yielding results at each step."""
        ask = self._ask(**kwargs)
        data = self._data(**kwargs)
        async for result in ask.__call_async__(goal, data, *args):
            yield result
    
    # Add this method to maintain compatibility with existing code
    def __aiter__(self, goal: str, *args, **kwargs):
        """Makes the object usable with 'async for' syntax.
        
        Example:
            async for result in df.ask.__aiter__("Analyze this data"):
                # Process result
        """
        return self.__call_async__(goal, *args, **kwargs)

    def code(self, goal: str, *args, **kwargs) -> None:
        """Print the generated code for the given goal."""
        ask = self._ask(**kwargs)
        data = self._data(**kwargs)
        return ask.code(goal, data, *args)

    def prompt(self, goal: str, *args, **kwargs) -> None:
        """Print the prompt that would be sent to the LLM."""
        ask = self._ask(**kwargs)
        data = self._data(**kwargs)
        return ask.prompt(goal, data, *args)