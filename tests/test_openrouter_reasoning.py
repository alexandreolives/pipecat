from pipecat.services.openrouter.llm import OpenRouterLLMService, OpenRouterLLMSettings


def test_openrouter_reasoning_is_moved_to_extra_body():
    service = OpenRouterLLMService(
        api_key="test-api-key",
        settings=OpenRouterLLMSettings(
            model="openai/gpt-4.1",
            extra={"reasoning": {"effort": "none"}},
        ),
    )

    params = service.build_chat_completion_params({"messages": []})

    assert "reasoning" not in params
    assert params["extra_body"] == {"reasoning": {"effort": "none"}}
