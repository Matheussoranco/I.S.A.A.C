class PromptLoader:
    @staticmethod
    def get(name: str, **kwargs) -> str:
        # Fallback to simple strings if file missing, but design for file-based
        prompts = {
            "system_default": "You are I.S.A.A.C., a local-first autonomous agent...",
            "researcher": "You are in Researcher mode. Focus on evidence and citations...",
            "operator": "You are in Operator mode. Focus on execution and efficiency...",
        }
        text = prompts.get(name, "")
        for k, v in kwargs.items():
            text = text.replace(f"{{ {k} }}", str(v))
        return text
