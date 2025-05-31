"""
TTS Engine for Google Gemini TTS.
"""
import json
import logging
import time
import base64

from google import genai
from google.genai import types
from homeassistant.exceptions import HomeAssistantError

_LOGGER = logging.getLogger(__name__)


class AudioResponse:
    """A simple response wrapper with a 'content' attribute to hold audio bytes."""
    def __init__(self, content: bytes):
        self.content = content


class OpenAITTSEngine:
    """
    A TTS engine that uses Google Gemini's native TTS capabilities instead of OpenAI.
    
    Note:
      - `model` should be set to one of the Gemini TTS model names, e.g., "gemini-2.5-flash-preview-tts".
      - `voice` should be one of the supported Gemini voice names (e.g., "Kore", "Puck", etc.).
      - The `url` parameter is no longer used by Gemini TTS but is retained here for backward compatibility.
      - The `speed` parameter is not directly supported by the Gemini TTS API. If you wish to alter pacing
        you can include instructions in the `instructions` argument (e.g., "Speak slowly and clearly: ...").
    """

    def __init__(self, api_key: str, voice: str, model: str, speed: float = 1.0, url: str = None):
        """
        :param api_key:       Your Google Cloud API key for Gemini.
        :param voice:         A prebuilt Gemini voice name (e.g., "Kore", "Puck", "Desina", etc.).
        :param model:         The Gemini TTS model to use (e.g., "gemini-2.5-flash-preview-tts").
        :param speed:         Placeholder for compatibility; Gemini TTS does not accept a numeric speed parameter directly.
        :param url:           Retained for backward compatibility; not used by Gemini TTS.
        """
        self._api_key = api_key
        self._voice = voice
        self._model = model
        self._speed = speed
        self._url = url  # Not used for Gemini TTS, but kept for interface compatibility.

        # Initialize the Gemini client with the provided API key.
        try:
            self._client = genai.Client(api_key=self._api_key)
        except Exception as e:
            _LOGGER.exception("Failed to initialize Google Gemini client")
            raise HomeAssistantError("Invalid Gemini API key or client initialization failed") from e

    def get_tts(
        self,
        text: str,
        speed: float = None,
        instructions: str = None,
        voice: str = None
    ) -> AudioResponse:
        """
        Synchronous TTS request using Google Gemini TTS.
        If the API call fails, retries once after a 1-second delay.

        :param text:          The text to convert to speech.
        :param speed:         Ignored by Gemini TTS (Kept for compatibility).
        :param instructions:  Optional natural-language instructions to guide style/accent/pace.
                              If provided, these instructions will be prefixed to the text.
        :param voice:         Overrides the default voice set during initialization, if provided.
        :return:              AudioResponse containing raw audio bytes (WAV format).
        :raises HomeAssistantError: If network or API errors occur even after retrying.
        """
        # Determine which voice to use
        if voice is None:
            voice = self._voice

        # If instructions are provided, prepend them to the text prompt.
        # Example: instructions="Speak in a cheerful tone:"
        if instructions:
            prompt = f"{instructions}\n{text}"
        else:
            prompt = text

        # If the caller passed a speed value, log a warning: Gemini TTS does not accept a direct speed parameter.
        if speed is not None:
            _LOGGER.warning(
                "Gemini TTS does not support a numeric 'speed' parameter; ignoring speed=%s", speed
            )

        # Build the Gemini GenerateContentConfig for single-speaker TTS
        speech_config = types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
            )
        )
        generate_config = types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=speech_config
        )

        max_retries = 1
        attempt = 0

        while True:
            try:
                # Call Gemini to generate TTS audio
                response = self._client.models.generate_content(
                    model=self._model,
                    contents=prompt,
                    config=generate_config,
                )

                # The returned audio is base64-encoded in:
                #   response.candidates[0].content.parts[0].inlineData.data
                candidate = (
                    response.candidates[0]
                    if getattr(response, "candidates", None) and len(response.candidates) > 0
                    else None
                )
                if not candidate or not getattr(candidate.content, "parts", None):
                    raise HomeAssistantError("No audio returned from Gemini TTS")

                parts = candidate.content.parts
                if not parts or not getattr(parts[0], "inlineData", None):
                    raise HomeAssistantError("Malformed response from Gemini TTS")

                b64_data = parts[0].inlineData.data
                audio_bytes = base64.b64decode(b64_data)
                return AudioResponse(audio_bytes)

            except Exception as exc:
                _LOGGER.exception(
                    "Error in Gemini TTS get_tts on attempt %d: %s", attempt + 1, exc
                )
                if attempt < max_retries:
                    attempt += 1
                    time.sleep(1)
                    _LOGGER.debug("Retrying Gemini TTS call (attempt %d)", attempt + 1)
                    continue
                else:
                    raise HomeAssistantError(
                        "Failed to fetch TTS audio from Gemini after retries"
                    ) from exc

    def close(self):
        """Nothing to close for the Gemini TTS client."""
        pass

    @staticmethod
    def get_supported_langs() -> list:
        """
        Returns the list of BCP-47 language codes that Gemini TTS can auto-detect.
        Gemini supports 24 languages, but since Gemini auto-detects, we return the standard codes.
        """
        return [
            "ar-EG", "de-DE", "en-US", "es-US", "fr-FR", "hi-IN",
            "id-ID", "it-IT", "ja-JP", "ko-KR", "nl-NL", "pl-PL",
            "pt-BR", "ru-RU", "th-TH", "tr-TR", "vi-VN", "ro-RO",
            "uk-UA", "bn-BD", "en-IN", "mr-IN", "ta-IN", "te-IN"
        ]
