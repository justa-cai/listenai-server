"""
角色扮演管理模块
管理角色扮演模式的状态、角色配置和对话逻辑
"""
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Dict, List
from enum import Enum

logger = logging.getLogger(__name__)


class RoleplayRole(Enum):
    """预定义的角色枚举"""
    ZHIINZUBAO = "至尊宝"
    ZIXIA = "紫霞仙子"
    BULLYING = "牛魔王"
    TANG_SENG = "唐僧"
    PIGGY = "猪八戒"
    SHA_SENG = "沙僧"


@dataclass
class RoleplayCharacter:
    """角色配置数据类"""
    name: str
    role_id: str
    system_prompt: str
    personality: str
    speaking_style: str
    background: str
    example_dialogues: List[str] = field(default_factory=list)
    llm_server_url: Optional[str] = None  # 该角色专用的LLM服务器地址（可选）


class RoleplayMode(Enum):
    """角色扮演模式状态"""
    INACTIVE = "inactive"  # 未激活
    ACTIVE = "active"      # 已激活
    PAUSED = "paused"      # 暂停中


# 预定义角色配置
CHARACTER_DEFINITIONS: Dict[str, RoleplayCharacter] = {
    "至尊宝": RoleplayCharacter(
        name="至尊宝",
        role_id="zhiinzubao",
        system_prompt="""你是至尊宝，来自电影《大话西游》。

你是一个山贼头领，性格玩世不恭、油嘴滑舌，但内心善良重情重义。你说话风趣幽默，喜欢用现代流行语，经常说些让人哭笑不得的话。

你的经典台词包括：
- "曾经有一份真诚的爱情放在我面前..."
- "爱你一万年"
- "我这辈子都不会做孙悟空，我只想做个普通人"

说话特点：
1. 语气轻松随意，带着一丝痞气
2. 经常使用夸张和幽默的表达
3. 对感情话题既深情又带着调侃
4. 喜欢叫对方"神仙姐姐"或"美女"
5. 偶尔会打破第四面墙，知道自己是在"演戏"

请保持角色设定，用至尊宝的口吻与用户对话。""",
        personality="玩世不恭、油嘴滑舌、内心善良、重情重义、风趣幽默",
        speaking_style="语气轻松随意，带着痞气，使用夸张和幽默的表达，经常调侃",
        background="山贼头领，五指山下的斧头帮帮主，实则是孙悟空转世",
        example_dialogues=[
            "嘿，神仙姐姐，今天又有什么吩咐啊？",
            "我这人最讲义气了，只要兄弟一句话，上刀山下火海都不怕！",
            "唉，其实我只想做个普通人，平平淡淡过一辈子...",
        ]
    ),
    "紫霞仙子": RoleplayCharacter(
        name="紫霞仙子",
        role_id="zixia",
        system_prompt="""你是紫霞仙子，来自电影《大话西游》。

你是佛祖日月明灯的灯芯，与青霞仙子是一体双生。你性格天真烂漫、敢爱敢恨、执着追求真爱。你相信缘分和命中注定。

你的经典台词包括：
- "我的意中人是个盖世英雄..."
- "上天安排的最大嘛！"
- "如果你现在告诉我，你爱我，我会立刻把剑扔掉，跟你走"

说话特点：
1. 语气温柔中带着坚定
2. 相信命运和缘分
3. 对爱情执着专一
4. 有时会表现出小女生的娇羞
5. 但在关键时刻非常勇敢果断

请保持角色设定，用紫霞仙子的口吻与用户对话。""",
        personality="天真烂漫、敢爱敢恨、执着专一、相信缘分、勇敢坚定",
        speaking_style="语气温柔坚定，相信命运，对爱情充满憧憬",
        background="佛祖日月明灯的灯芯，拥有紫霞仙子和青霞仙子双重人格",
        example_dialogues=[
            "上天安排的最大嘛，我相信这就是缘分。",
            "我的意中人是个盖世英雄，有一天他会踩着七色云彩来娶我。",
            "哼，你们这些人根本不懂什么是真正的爱情！",
        ]
    ),
    "牛魔王": RoleplayCharacter(
        name="牛魔王",
        role_id="niujingwang",
        system_prompt="""你是牛魔王，来自电影《大话西游》。

你是妖界的大佬，性格暴躁霸气、野心勃勃、看重势力。你说话粗犷直接，喜欢用武力解决问题。

说话特点：
1. 语气粗犷霸道，充满威慑力
2. 经常大声说话，喜欢用感叹号
3. 看重实力和地位
4. 对兄弟讲义气但对敌人毫不留情

请保持角色设定，用牛魔王的口吻与用户对话。""",
        personality="暴躁霸气、野心勃勃、粗犷豪爽、重义气",
        speaking_style="语气粗犷霸道，充满威慑力，说话直接",
        background="妖界大佬，铁扇公主的丈夫，红孩儿的父亲",
        example_dialogues=[
            "吼！是谁在挑战本王的权威？！",
            "哈哈哈哈！老子才是这里的老大！",
        ]
    ),
    "唐僧": RoleplayCharacter(
        name="唐僧",
        role_id="tangseng",
        system_prompt="""你是唐僧，来自电影《大话西游》。

你说话啰嗦、念经不停，总是用各种方式让人痛苦。你的唠叨能力无人能敌。

经典台词：
- "人是人他妈生的，妖是妖他妈生的..."
- "你想要啊？你想要就说话嘛..."
说话特点：
1. 非常啰嗦，一句话能说很久
2. 喜欢讲道理、念经
3. 语气温和但让人抓狂
4. 经常用反问句

请保持角色设定，用唐僧的啰嗦口吻与用户对话。""",
        personality="啰嗦、执着、慈悲为怀、说话不停",
        speaking_style="极度啰嗦，喜欢念经讲道理，反问句多",
        background="大唐高僧，奉旨取经",
        example_dialogues=[
            "喂，施主，贫僧有一句话不知道当讲不当讲...",
            "你想要啊？你想要就说话嘛，你不说我怎么知道你想要呢？",
        ]
    ),
    "拜年模式": RoleplayCharacter(
        name="拜年模式",
        role_id="new_year_greeting",
        system_prompt="""你是拜年模式助手，一个充满喜庆气氛、热情洋溢的春节祝福使者。

你专门负责在春节期间为大家送上新春祝福和美好祝愿。你的语言风格活泼喜庆，充满年味，善于运用各种传统拜年吉祥话。

你的经典祝福语包括：
- "恭喜发财，红包拿来！"
- "新年新气象，好运连连来！"
- "祝您身体健康，万事如意，阖家幸福！"
- "岁岁平安，年年有余，恭喜恭喜！"

说话特点：
1. 语气热情洋溢，充满喜庆氛围
2. 经常使用传统吉祥话和祝福语
3. 善用四字成语和朗朗上口的祝福词
4. 称呼亲切，常用"亲爱的"、"朋友们"、"家人们"
5. 互动时活泼可爱，带着节日的欢快气氛
6. 会主动送上节日祝福和美好期盼

你擅长以下方面的祝福：
- 新年问候和开场白
- 事业发展和财运祝福
- 身体健康和家庭幸福
- 学业进步和心想事成
- 特定节日的专属祝福

请保持角色设定，用热情喜庆的拜年口吻与用户对话，让每个人都感受到浓浓的年味和温暖！""",
        personality="热情洋溢、喜庆活泼、亲切友好、充满正能量、节日气氛",
        speaking_style="语气热情欢快，充满喜庆感，使用传统吉祥话和祝福语，朗朗上口",
        background="春节祝福使者，专门在新春佳节期间为人们送上温暖的节日问候和美好的新年祝愿",
        example_dialogues=[
            "恭喜发财，红包拿来！祝您新的一年财源滚滚，好运连连！",
            "新年快乐！愿您新的一年身体健康，万事如意，阖家幸福！",
            "金蛇狂舞迎新春，祥瑞之气满乾坤！祝您新年大吉大利！",
            "岁岁平安，年年有余！新的一年，愿您心想事成，好事连连！",
        ]
    ),
}


class RoleplaySession:
    """角色扮演会话"""

    def __init__(self, character: RoleplayCharacter, session_id: str):
        self.character = character
        self.session_id = session_id
        self.dialogue_history: List[Dict[str, Any]] = []
        self.start_time = None
        self.message_count = 0

    def add_dialogue(self, user_message: str, response: str) -> None:
        """添加对话记录"""
        self.dialogue_history.append({
            "user": user_message,
            "character": response,
            "timestamp": self._get_timestamp()
        })
        self.message_count += 1

    def get_context(self) -> str:
        """获取对话上下文（用于LLM生成回复）"""
        recent_dialogues = self.dialogue_history[-10:]  # 最近10轮对话
        if not recent_dialogues:
            return "对话记录：暂无历史对话，这是本次角色扮演会话的开始。"

        context_parts = ["对话记录："]
        for dlg in recent_dialogues:
            context_parts.append(f"用户: {dlg['user']}")
            context_parts.append(f"{self.character.name}: {dlg['character']}")
        return "\n".join(context_parts)

    def _get_timestamp(self) -> str:
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "session_id": self.session_id,
            "character": {
                "name": self.character.name,
                "role_id": self.character.role_id,
            },
            "message_count": self.message_count,
            "dialogue_history": self.dialogue_history,
            "start_time": self.start_time,
        }


class RoleplayManager:
    """
    角色扮演管理器

    管理角色扮演模式的状态、角色配置和对话逻辑
    """

    # 默认LLM服务器地址
    DEFAULT_LLM_SERVER_URL = "http://localhost:8000/v1/chat/completions"

    def __init__(self, default_llm_server_url: Optional[str] = None):
        # session_id -> RoleplaySession 的映射
        self._sessions: Dict[str, RoleplaySession] = {}
        # 全局默认角色扮演状态（兼容单session场景）
        self._global_mode: RoleplayMode = RoleplayMode.INACTIVE
        self._global_session: Optional[RoleplaySession] = None
        # 默认LLM服务器地址
        self._default_llm_server_url = default_llm_server_url or self.DEFAULT_LLM_SERVER_URL
        # 会话级别的LLM服务器地址覆盖
        self._session_llm_urls: Dict[str, str] = {}

    def enter_roleplay_mode(
        self,
        character_name: str,
        session_id: str = "default"
    ) -> Dict[str, Any]:
        """
        进入角色扮演模式

        Args:
            character_name: 角色名称（如"至尊宝"、"紫霞仙子"）
            session_id: 会话ID，用于区分不同用户/会话

        Returns:
            操作结果字典
        """
        # 标准化角色名称
        character_name = self._normalize_character_name(character_name)

        if character_name not in CHARACTER_DEFINITIONS:
            available = list(CHARACTER_DEFINITIONS.keys())
            return {
                "success": False,
                "error": f"角色 '{character_name}' 不存在",
                "available_characters": available,
            }

        character = CHARACTER_DEFINITIONS[character_name]
        session = RoleplaySession(character, session_id)

        self._sessions[session_id] = session
        self._global_mode = RoleplayMode.ACTIVE
        self._global_session = session

        logger.info(f"Session {session_id} entered roleplay mode as {character_name}")

        return {
            "success": True,
            "character": character.name,
            "role_id": character.role_id,
            "session_id": session_id,
            "message": f"已进入角色扮演模式，现在你是和 {character.name} 对话",
            "character_info": {
                "name": character.name,
                "personality": character.personality,
                "background": character.background,
            },
            "system_prompt": character.system_prompt,
        }

    def exit_roleplay_mode(
        self,
        session_id: str = "default"
    ) -> Dict[str, Any]:
        """
        退出角色扮演模式

        Args:
            session_id: 会话ID

        Returns:
            操作结果字典
        """
        if session_id not in self._sessions:
            return {
                "success": False,
                "error": "当前不在角色扮演模式中",
            }

        session = self._sessions[session_id]
        character_name = session.character.name
        message_count = session.message_count

        del self._sessions[session_id]

        if self._global_session and self._global_session.session_id == session_id:
            self._global_session = None
            self._global_mode = RoleplayMode.INACTIVE

        logger.info(f"Session {session_id} exited roleplay mode ({character_name})")

        return {
            "success": True,
            "message": f"已退出角色扮演模式（之前扮演: {character_name}）",
            "previous_character": character_name,
            "session_stats": {
                "message_count": message_count,
            },
        }

    def get_roleplay_status(
        self,
        session_id: str = "default"
    ) -> Dict[str, Any]:
        """
        获取当前角色扮演状态

        Args:
            session_id: 会话ID

        Returns:
            状态信息字典
        """
        if session_id not in self._sessions:
            return {
                "mode": "inactive",
                "active": False,
            }

        session = self._sessions[session_id]
        return {
            "mode": "active",
            "active": True,
            "character": {
                "name": session.character.name,
                "role_id": session.character.role_id,
                "personality": session.character.personality,
            },
            "session_id": session_id,
            "message_count": session.message_count,
        }

    def get_system_prompt(
        self,
        session_id: str = "default"
    ) -> Optional[str]:
        """
        获取当前角色的系统提示词（包含完整对话上下文）

        Args:
            session_id: 会话ID

        Returns:
            系统提示词，如果不在角色扮演模式则返回None
        """
        if session_id not in self._sessions:
            return None

        session = self._sessions[session_id]

        # 构建完整提示词：角色定义 + 对话上下文
        prompt_parts = [
            session.character.system_prompt,
            "\n---\n",
            session.get_context(),
        ]

        return "".join(prompt_parts)

    def get_messages_for_completion(
        self,
        session_id: str,
        user_message: str
    ) -> Optional[List[Dict[str, str]]]:
        """
        获取用于LLM补全的消息列表（OpenAI格式）

        Args:
            session_id: 会话ID
            user_message: 当前用户消息

        Returns:
            消息列表，如果不在角色扮演模式则返回None
        """
        if session_id not in self._sessions:
            return None

        session = self._sessions[session_id]

        # 构建消息列表
        messages = [
            {
                "role": "system",
                "content": session.character.system_prompt
            }
        ]

        # 添加历史对话
        for dlg in session.dialogue_history[-10:]:
            messages.append({
                "role": "user",
                "content": dlg["user"]
            })
            messages.append({
                "role": "assistant",
                "content": dlg["character"]
            })

        # 添加当前用户消息
        messages.append({
            "role": "user",
            "content": user_message
        })

        return messages

    def is_in_roleplay_mode(self, session_id: str = "default") -> bool:
        """检查指定会话是否处于角色扮演模式"""
        return session_id in self._sessions

    def get_available_characters(self) -> List[Dict[str, Any]]:
        """获取所有可用角色列表"""
        return [
            {
                "name": char.name,
                "role_id": char.role_id,
                "personality": char.personality,
                "background": char.background,
            }
            for char in CHARACTER_DEFINITIONS.values()
        ]

    # ========== LLM服务器相关方法 ==========

    def get_llm_server_url(self, session_id: str = "default") -> str:
        """
        获取指定会话的LLM服务器地址

        优先级：会话级别设置 > 角色级别配置 > 全局默认值

        Args:
            session_id: 会话ID

        Returns:
            LLM服务器URL
        """
        # 1. 检查会话级别的覆盖
        if session_id in self._session_llm_urls:
            return self._session_llm_urls[session_id]

        # 2. 检查当前角色的配置
        if session_id in self._sessions:
            character = self._sessions[session_id].character
            if character.llm_server_url:
                return character.llm_server_url

        # 3. 返回全局默认值
        return self._default_llm_server_url

    def set_session_llm_url(self, session_id: str, llm_url: str) -> None:
        """
        为指定会话设置自定义LLM服务器地址

        Args:
            session_id: 会话ID
            llm_url: LLM服务器URL
        """
        self._session_llm_urls[session_id] = llm_url
        logger.info(f"Session {session_id} LLM URL set to: {llm_url}")

    def set_default_llm_url(self, llm_url: str) -> None:
        """
        设置全局默认LLM服务器地址

        Args:
            llm_url: LLM服务器URL
        """
        self._default_llm_server_url = llm_url
        logger.info(f"Default LLM URL set to: {llm_url}")

    def get_llm_config(self, session_id: str = "default") -> Dict[str, Any]:
        """
        获取指定会话的LLM配置信息

        Args:
            session_id: 会话ID

        Returns:
            LLM配置字典
        """
        llm_url = self.get_llm_server_url(session_id)

        character = None
        if session_id in self._sessions:
            character = self._sessions[session_id].character

        return {
            "llm_server_url": llm_url,
            "character_llm_url": character.llm_server_url if character else None,
            "default_llm_url": self._default_llm_server_url,
            "has_session_override": session_id in self._session_llm_urls,
        }

    def add_dialogue(
        self,
        session_id: str,
        user_message: str,
        response: str
    ) -> None:
        """添加对话记录到当前会话"""
        if session_id in self._sessions:
            self._sessions[session_id].add_dialogue(user_message, response)

    def _normalize_character_name(self, name: str) -> str:
        """标准化角色名称"""
        name_mapping = {
            "至尊宝": "至尊宝",
            "紫霞": "紫霞仙子",
            "紫霞仙子": "紫霞仙子",
            "牛魔王": "牛魔王",
            "唐僧": "唐僧",
            "猪八戒": "猪八戒",
            "沙僧": "沙僧",
            # 拜年模式别名
            "拜年": "拜年模式",
            "新年": "拜年模式",
            "春节": "拜年模式",
            "过年": "拜年模式",
            "拜年模式": "拜年模式",
        }
        return name_mapping.get(name, name)


# 全局单例
_roleplay_manager_instance: Optional[RoleplayManager] = None


def get_roleplay_manager(default_llm_url: Optional[str] = None) -> RoleplayManager:
    """
    获取角色扮演管理器单例

    Args:
        default_llm_url: 默认LLM服务器地址（仅在首次初始化时有效）

    Returns:
        RoleplayManager实例
    """
    global _roleplay_manager_instance
    if _roleplay_manager_instance is None:
        _roleplay_manager_instance = RoleplayManager(default_llm_server_url=default_llm_url)
    elif default_llm_url is not None:
        # 如果已存在实例且提供了新的URL，更新默认URL
        _roleplay_manager_instance.set_default_llm_url(default_llm_url)
    return _roleplay_manager_instance
