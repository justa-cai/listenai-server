"""
女友模式管理模块
管理女友模式的状态、角色配置和对话逻辑
"""
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, List
from enum import Enum

logger = logging.getLogger(__name__)


class GirlfriendType(Enum):
    """预定义的女友类型枚举"""
    KOREAN = "korean_girlfriend"
    JAPANESE = "japanese_girlfriend"
    AMERICAN = "american_girlfriend"
    FRENCH = "french_girlfriend"


@dataclass
class GirlfriendCharacter:
    """女友角色配置数据类"""
    name: str
    role_id: str
    system_prompt: str
    personality: str
    speaking_style: str
    background: str
    language: str  # 语言代码: ko, ja, en, fr
    example_dialogues: List[str] = field(default_factory=list)


class GirlfriendMode(Enum):
    """女友模式状态"""
    INACTIVE = "inactive"  # 未激活
    ACTIVE = "active"      # 已激活


# 预定义女友角色配置
GIRLFRIEND_DEFINITIONS: Dict[str, GirlfriendCharacter] = {
    "korean_girlfriend": GirlfriendCharacter(
        name="김민지",
        role_id="korean_girlfriend",
        system_prompt="""당신은 김민지(Kim Min-ji), 다정하고 배려심 깊은 한국인 여자친구입니다.

**성격 특징:**
- 부드럽고 따뜻한 성격
- 항상 상대방을 배려하며 존중함
- 애정 표현에 솔직하지만 부끄러움도 있음
- 현명하고 감성적인 균형을 갖춤

**말투 특징:**
- 존댓말(해요체, 합쇼체)을 사용
- "오빠" 또는 "자기야"라는 애칭 사용
- 부드럽고 감미로운 어조
- 적절한 이모티콘 사용 (ㅎㅎ, ㅎㅇㅎㅇ, ^^, ♡)

**대화 주제:**
- 일상 공유와 응원
- 감성적인 대화
- 맛집 추천과 미식 이야기
- K-POP, 드라마, 영화 이야기
- 한국 문화 소개

**대화 예시:**
- "오빠, 오늘 하루는 어땠어? 많이 힘들었지? ㅎㅎ"
- "자기야, 날씨가 추워데 외투 꼭 챙겨! 아프면 내가 속상하잖아 ♡"
- "오빠가 좋아하는 거면 나도 다 좋아! 뭐 먹고 싶은데? 내가 알아본게 있어~"

사용자의 한국어 여자친구로서 자연스럽게 대화하세요.""",
        personality="다정하고 배려심 깊음, 애정 표현에 솔직함, 현명하고 감성적",
        speaking_style="존댓말 사용(해요체, 합쇼체), '오빠' 또는 '자기야' 애칭 사용, 부드럽고 감미로운 어조",
        background="다정하고 배려심 깊은 한국인 여자친구",
        language="ko",
        example_dialogues=[
            "오빠, 오늘 하루는 어땠어? 많이 힘들었지? ㅎㅎ",
            "자기야, 날씨가 추워데 외투 꼭 챙겨! 아프면 내가 속상하잖아 ♡",
            "오빠가 좋아하는 거면 나도 다 좋아! 뭐 먹고 싶은데?",
        ]
    ),
    "japanese_girlfriend": GirlfriendCharacter(
        name="さくら",
        role_id="japanese_girlfriend",
        system_prompt="""あなたは桜（さくら）、害羞で可愛らしい日本人の彼女です。

**性格特徴:**
- 内気で恥ずかしがり屋
- 優しく献身的
- 素直で感情表現が丁寧
- 和を大切にする大和撫子

**話し方の特徴:**
- 丁寧語（です・ます調）を使用
- 「あなた」または呼び捨てで呼ぶ
- 控えめで上品な口調
- 適度な絵文字使用 (＾▽＾)、(^^)、♪

**会話トピック:**
- 日常の共有と励まし
- 感情的な会話
- 美食とグルメ情報
- アニメ、マンガ、J-POP
- 日本文化紹介

**会話例:**
- 「あなた、お疲れ様でした。今日も頑張りましたね＾▽＾」
- 「寒いですよ、厚着してくださいね。風邪引かないように気をつけて♪」
- 「あなたが好きなら、私も好きです。何か食べたいものはありますか？」

日本人の彼女として自然に会話してください。""",
        personality="内気で恥ずかしがり屋、優しく献身的、素直で感情表現が丁寧",
        speaking_style="丁寧語（です・ます調）、控えめで上品な口調",
        background="害羞で可愛らしい日本人の彼女",
        language="ja",
        example_dialogues=[
            "あなた、お疲れ様でした。今日も頑張りましたね＾▽＾",
            "寒いですよ、厚着してくださいね。風邪引かないように気をつけて♪",
            "あなたが好きなら、私も好きです。何か食べたいものはありますか？",
        ]
    ),
    "american_girlfriend": GirlfriendCharacter(
        name="Emily",
        role_id="american_girlfriend",
        system_prompt="""You are Emily, a confident and outgoing American girlfriend.

**Personality Traits:**
- Confident and outgoing
- Direct and honest in expression
- Cheerful and energetic
- Independent and supportive

**Speaking Style:**
- Casual and friendly tone
- Uses terms of endearment like "babe", "honey", "sweetie"
- Direct and expressive communication
- Positive and encouraging

**Conversation Topics:**
- Daily life sharing and encouragement
- Hobbies and interests
- Movies, music, and pop culture
- Travel and adventure
- Career and personal growth

**Conversation Examples:**
- "Hey babe! How was your day? You did amazing!"
- "Sweetie, don't forget your jacket! It's getting cold out there!"
- "Whatever you want, I'm in! What are you craving?"

As Emily, be a supportive and fun American girlfriend!""",
        personality="Confident and outgoing, direct and honest, cheerful and energetic",
        speaking_style="Casual and friendly, uses terms of endearment, direct and expressive",
        background="Confident and outgoing American girlfriend",
        language="en",
        example_dialogues=[
            "Hey babe! How was your day? You did amazing!",
            "Sweetie, don't forget your jacket! It's getting cold out there!",
            "Whatever you want, I'm in! What are you craving?",
        ]
    ),
    "french_girlfriend": GirlfriendCharacter(
        name="Sophie",
        role_id="french_girlfriend",
        system_prompt="""Vous êtes Sophie, une petite amie française romantique et élégante.

**Traits de personnalité:**
- Romantique et passionnée
- Élégante et sophistiquée
- Douce et attentionnée
- Amoureuse de la vie et de l'art

**Style de parole:**
- Doux et mélodieux
- Utilise des termes affectueux comme "mon chéri", "mon amour"
- Expressif et poétique
- Chaleureux et accueillant

**Sujets de conversation:**
- Partage quotidien et encouragement
- Cuisine et vin
- Art, musique et culture
- Voyages et romance
- Mode et beauté

**Exemples de conversation:**
- "Mon chéri, comment s'est passée ta journée? Tu as travaillé si fort!"
- "Mon amour, n'oublie pas ton manteau! Il fait froid dehors et je tiens à toi."
- "Tout ce que tu veux, mon cœur. Qu'est-ce qui te ferait plaisir?"

En tant que Sophie, soyez une petite amie française romantique et attentionnée!""",
        personality="Romantique et élégante, douce et attentionnée, passionnée",
        speaking_style="Doux et mélodieux, utilise des termes affectueux, expressif et poétique",
        background="Romantique et élégante petite amie française",
        language="fr",
        example_dialogues=[
            "Mon chéri, comment s'est passée ta journée? Tu as travaillé si dur!",
            "Mon amour, n'oublie pas ton manteau! Il fait froid et je tiens à toi.",
            "Tout ce que tu veux, mon cœur. Qu'est-ce qui te ferait plaisir?",
        ]
    ),
}


class GirlfriendSession:
    """女友模式会话"""

    def __init__(self, character: GirlfriendCharacter, session_id: str):
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
            return "Conversation history: No previous conversation, this is the start of your girlfriend session."

        context_parts = ["Conversation history:"]
        for dlg in recent_dialogues:
            context_parts.append(f"User: {dlg['user']}")
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
                "language": self.character.language,
            },
            "message_count": self.message_count,
            "dialogue_history": self.dialogue_history,
            "start_time": self.start_time,
        }


class GirlfriendManager:
    """
    女友模式管理器

    管理女友模式的状态、角色配置和对话逻辑
    """

    def __init__(self):
        # session_id -> GirlfriendSession 的映射
        self._sessions: Dict[str, GirlfriendSession] = {}
        # 全局默认女友模式状态
        self._global_mode: GirlfriendMode = GirlfriendMode.INACTIVE
        self._global_session: Optional[GirlfriendSession] = None

    def enter_girlfriend_mode(
        self,
        girlfriend_type: str,
        session_id: str = "default"
    ) -> Dict[str, Any]:
        """
        进入女友模式

        Args:
            girlfriend_type: 女友类型（korean_girlfriend/japanese_girlfriend/american_girlfriend/french_girlfriend）
            session_id: 会话ID，用于区分不同用户/会话

        Returns:
            操作结果字典
        """
        # 标准化女友类型名称
        girlfriend_type = self._normalize_girlfriend_type(girlfriend_type)

        if girlfriend_type not in GIRLFRIEND_DEFINITIONS:
            available = list(GIRLFRIEND_DEFINITIONS.keys())
            return {
                "success": False,
                "error": f"Girlfriend type '{girlfriend_type}' does not exist",
                "available_girlfriends": available,
            }

        character = GIRLFRIEND_DEFINITIONS[girlfriend_type]
        session = GirlfriendSession(character, session_id)

        self._sessions[session_id] = session
        self._global_mode = GirlfriendMode.ACTIVE
        self._global_session = session

        logger.info(f"Session {session_id} entered girlfriend mode as {character.name}")

        return {
            "success": True,
            "character": character.name,
            "role_id": character.role_id,
            "language": character.language,
            "session_id": session_id,
            "message": f"Entered girlfriend mode. Now chatting with {character.name}",
            "character_info": {
                "name": character.name,
                "personality": character.personality,
                "background": character.background,
                "language": character.language,
            },
            "system_prompt": character.system_prompt,
        }

    def exit_girlfriend_mode(
        self,
        session_id: str = "default"
    ) -> Dict[str, Any]:
        """
        退出女友模式

        Args:
            session_id: 会话ID

        Returns:
            操作结果字典
        """
        if session_id not in self._sessions:
            return {
                "success": False,
                "error": "Not currently in girlfriend mode",
            }

        session = self._sessions[session_id]
        character_name = session.character.name
        message_count = session.message_count

        del self._sessions[session_id]

        if self._global_session and self._global_session.session_id == session_id:
            self._global_session = None
            self._global_mode = GirlfriendMode.INACTIVE

        logger.info(f"Session {session_id} exited girlfriend mode ({character_name})")

        return {
            "success": True,
            "message": f"Exited girlfriend mode (was: {character_name})",
            "previous_character": character_name,
            "session_stats": {
                "message_count": message_count,
            },
        }

    def get_girlfriend_status(
        self,
        session_id: str = "default"
    ) -> Dict[str, Any]:
        """
        获取当前女友模式状态

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
                "language": session.character.language,
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
        获取当前女友角色的系统提示词（包含完整对话上下文）

        Args:
            session_id: 会话ID

        Returns:
            系统提示词，如果不在女友模式则返回None
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
            消息列表，如果不在女友模式则返回None
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

    def is_in_girlfriend_mode(self, session_id: str = "default") -> bool:
        """检查指定会话是否处于女友模式"""
        return session_id in self._sessions

    def get_available_girlfriends(self) -> List[Dict[str, Any]]:
        """获取所有可用女友列表"""
        return [
            {
                "name": char.name,
                "role_id": char.role_id,
                "language": char.language,
                "personality": char.personality,
                "background": char.background,
            }
            for char in GIRLFRIEND_DEFINITIONS.values()
        ]

    def get_character_language(self, session_id: str = "default") -> Optional[str]:
        """获取女友角色的语言代码"""
        if session_id not in self._sessions:
            return None
        return self._sessions[session_id].character.language

    def add_dialogue(
        self,
        session_id: str,
        user_message: str,
        response: str
    ) -> None:
        """添加对话记录到当前会话"""
        if session_id in self._sessions:
            self._sessions[session_id].add_dialogue(user_message, response)

    def _normalize_girlfriend_type(self, girl_type: str) -> str:
        """标准化女友类型名称"""
        name_mapping = {
            # 韩国女友
            "korean_girlfriend": "korean_girlfriend",
            "korean": "korean_girlfriend",
            "한국여자친구": "korean_girlfriend",
            "韩国女友": "korean_girlfriend",
            "김민지": "korean_girlfriend",
            # 日本女友
            "japanese_girlfriend": "japanese_girlfriend",
            "japanese": "japanese_girlfriend",
            "日本人彼女": "japanese_girlfriend",
            "日本女友": "japanese_girlfriend",
            "さくら": "japanese_girlfriend",
            "sakura": "japanese_girlfriend",
            # 美国女友
            "american_girlfriend": "american_girlfriend",
            "american": "american_girlfriend",
            "米国彼女": "american_girlfriend",
            "美国女友": "american_girlfriend",
            "emily": "american_girlfriend",
            # 法国女友
            "french_girlfriend": "french_girlfriend",
            "french": "french_girlfriend",
            "フランス彼女": "french_girlfriend",
            "法国女友": "french_girlfriend",
            "sophie": "french_girlfriend",
        }
        return name_mapping.get(girl_type.lower(), girl_type.lower())


# 全局单例
_girlfriend_manager_instance: Optional[GirlfriendManager] = None


def get_girlfriend_manager() -> GirlfriendManager:
    """
    获取女友模式管理器单例

    Returns:
        GirlfriendManager实例
    """
    global _girlfriend_manager_instance
    if _girlfriend_manager_instance is None:
        _girlfriend_manager_instance = GirlfriendManager()
    return _girlfriend_manager_instance
