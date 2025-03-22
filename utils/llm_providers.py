from abc import ABC, abstractmethod


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

