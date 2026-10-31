"""Fast local voice presets; age labels describe an experimental sound style."""
from __future__ import annotations

TOKEN_ROOT = r"HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech_OneCore\Voices\Tokens\MSTTS_V110_zhCN_"
PRESETS = {
    "default": ("系统默认中文声", "", 0, 0),
    "teen_male": ("少年男声", "KangkangM", 1, 4),
    "teen_female": ("少年女声", "YaoyaoM", 1, 3),
    "young_male": ("青年男声", "KangkangM", 1, 1),
    "young_female": ("青年女声", "YaoyaoM", 0, 1),
    "middle_male": ("中年男声", "KangkangM", 0, -1),
    "middle_female": ("中年女声", "HuihuiM", -1, -1),
    "senior_male": ("老年男声", "KangkangM", -2, -3),
    "senior_female": ("老年女声", "HuihuiM", -2, -3),
}


def preset_profile(identity):
    from wuwa_dubber.models import VoiceProfile
    if identity not in PRESETS:
        raise ValueError(f"未知的系统声线：{identity}")
    label, token, rate, pitch = PRESETS[identity]
    return VoiceProfile(
        "系统中文朗读" if identity == "default" else "系统·" + label,
        speed=2 ** (rate / 10), system_voice=TOKEN_ROOT + token if token else "",
        system_pitch=pitch,
    )


class SystemVoiceRouter:
    def __init__(self, options, profiles=()):
        from wuwa_dubber.text_processing import normalize_speaker
        self.normalize = normalize_speaker
        self.default_id = options.get("system_default_voice", "default")
        self.default = preset_profile(self.default_id)
        self.bindings = {}
        aliases = {}
        for profile in profiles:
            for alias in [profile.speaker, *profile.aliases]:
                aliases[normalize_speaker(alias)] = normalize_speaker(profile.speaker)
        self.aliases = aliases
        for name, preset in options.get("system_character_voices", {}).items():
            key = normalize_speaker(name)
            if not key:
                raise ValueError("系统声线的角色名不能为空")
            key = aliases.get(key, key)
            voice = preset_profile(preset)
            if key in self.bindings and self.bindings[key] != voice:
                raise ValueError(f"角色和别名指定了不同声音：{name}")
            self.bindings[key] = voice

    def select(self, record):
        name = ""
        if record.get("speaker_status") == "verified":
            name = record.get("speaker", "")
        elif record.get("speaker_identity_confirmed"):
            name = record.get("speaker_identity", "")
        key = self.normalize(name)
        key = self.aliases.get(key, key)
        if key and key in self.bindings:
            return self.bindings[key], False
        return self.default, True

    def check_available(self, health):
        installed = {item["id"].casefold() for item in health.get("voices", [])}
        for profile in [self.default, *self.bindings.values()]:
            if profile.system_voice and profile.system_voice.casefold() not in installed:
                raise ValueError(f"本机未安装{profile.speaker}所需的中文声音，请在系统声线设置中选择其他声音")
