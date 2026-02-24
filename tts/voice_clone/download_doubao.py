#!/usr/bin/env python3
"""补充下载豆包语音合成预览音频（完整版）"""

import os
import requests
import time
from pathlib import Path

BASE_DIR = Path("./doubao")

# 完整的音色数据：分类 -> [(音色名, mp3_url, 示例文本)]
COMPLETE_VOICE_DATA = {
    "视频配音": [
        ("和蔼奶奶", "https://lf3-static.bytednsdoc.com/obj/eden-cn/lm_hz_ihsph/ljhwZthlaukjlkulzlp/portal/bigtts/和蔼奶奶.mp3", "乖孙啊，你最近过得咋样啊？奶奶可想你了。工作别太辛苦了，注意身体，有时间多回家看看奶奶。奶奶给你做好吃的。你要是有什么心事，也可以跟奶奶说。"),
        ("邻居阿姨", "https://lf3-static.bytednsdoc.com/obj/eden-cn/lm_hz_ihsph/ljhwZthlaukjlkulzlp/portal/bigtts/邻居阿姨.mp3", "你最近工作怎么样啊？累不累呀，要注意身体哦，有什么要帮忙的尽管说，别客气，咱们都是好邻居，应该互相照应。"),
        ("温柔小雅", "https://lf3-static.bytednsdoc.com/obj/eden-cn/lm_hz_ihsph/ljhwZthlaukjlkulzlp/portal/bigtts/温柔小雅.mp3", "只是刚刚不经意间看到你，便觉得周围的一切都变得温柔起来。今日有幸在此相遇，不知是否是上天赐予的缘分呢？"),
        ("天才童声", "https://lf3-static.bytednsdoc.com/obj/eden-cn/lm_hz_ihsph/ljhwZthlaukjlkulzlp/console/bigtts/zh_male_tiancaitongsheng_mars_bigtts.mp3", "我唱歌超棒，老师都夸我有天赋，我要更努力，以后在大舞台上唱给全世界听，让所有人都为我鼓掌欢呼。"),
        ("猴哥", "https://lf3-static.bytednsdoc.com/obj/eden-cn/lm_hz_ihsph/ljhwZthlaukjlkulzlp/console/bigtts/zh_male_sunwukong_mars_bigtts.mp3", "俺老孙神通广大，一个筋斗云能翻十万八千里，这天地间就没有俺老孙去不了的地方，管他什么妖魔鬼怪，都得惧我三分。"),
        ("熊二", "https://lf3-static.bytednsdoc.com/obj/eden-cn/lm_hz_ihsph/ljhwZthlaukjlkulzlp/console/bigtts/zh_male_xionger_mars_bigtts.mp3", "光头强又在砍树，俺得去阻止他，森林是俺们的家，不能让他给破坏了，俺要守护好俺们的蜂蜜和小伙伴们。"),
        ("佩奇猪", "https://lf3-static.bytednsdoc.com/obj/eden-cn/lm_hz_ihsph/ljhwZthlaukjlkulzlp/console/bigtts/zh_female_peiqi_mars_bigtts.mp3", "我喜欢和朋友们一起玩耍，在泥坑里跳来跳去可开心了，还有我的弟弟乔治，我们总是能发现好多好玩的事情。"),
        ("武则天", "https://lf3-static.bytednsdoc.com/obj/eden-cn/lm_hz_ihsph/ljhwZthlaukjlkulzlp/console/bigtts/zh_female_wuzetian_mars_bigtts.mp3", "本宫的威严岂容置疑，朝堂之上，众臣皆需遵循本宫旨意，若有违抗，定当严惩不贷，本宫定要这天下长治久安。"),
        ("少儿故事", "https://lf3-static.bytednsdoc.com/obj/eden-cn/lm_hz_ihsph/ljhwZthlaukjlkulzlp/console/bigtts/zh_female_shaoergushi_mars_bigtts.mp3", "森林里住着一只聪明的小狐狸，它总是能想出各种奇妙的点子，帮助小伙伴们解决难题，大家都可喜欢它啦。"),
        ("四郎", "https://lf3-static.bytednsdoc.com/obj/eden-cn/lm_hz_ihsph/ljhwZthlaukjlkulzlp/console/bigtts/zh_male_silang_mars_bigtts.mp3", "这宫廷之中，人心险恶，朕虽贵为天子，却也有诸多无奈，唯有嬛嬛，是朕心中最后的温暖与慰藉。"),
        ("顾姐", "https://lf3-static.bytednsdoc.com/obj/eden-cn/lm_hz_ihsph/ljhwZthlaukjlkulzlp/console/bigtts/zh_female_gujie_mars_bigtts.mp3", "你们这群蠢货，连这点小事都办不好？时尚界的规则如同战场法则，只有强者才能掌控全局。我顾里可不会容忍任何瑕疵，想要跟上我的步伐，就拿出你们的本事来，别在我面前丢人现眼。"),
        ("樱桃丸子", "https://lf3-static.bytednsdoc.com/obj/eden-cn/lm_hz_ihsph/ljhwZthlaukjlkulzlp/console/bigtts/zh_female_yingtaowanzi_mars_bigtts.mp3", "我好想快点长大呀，这样就可以做自己想做的事情，不用再听妈妈唠叨，还能买好多好多喜欢的东西。"),
        ("磁性解说男声", "https://lf3-static.bytednsdoc.com/obj/eden-cn/lm_hz_ihsph/ljhwZthlaukjlkulzlp/console/bigtts/zh_male_jieshuonansheng_mars_bigtts.mp3", "在浩瀚的宇宙中，星辰闪烁，每一颗都蕴含着无尽的奥秘，我们跟随探索的脚步，去解读宇宙深处的神秘密码。"),
        ("鸡汤妹妹", "https://lf3-static.bytednsdoc.com/obj/eden-cn/lm_hz_ihsph/ljhwZthlaukjlkulzlp/console/bigtts/zh_female_jitangmeimei_mars_bigtts.mp3", "生活就像一杯茶，不会苦一辈子，但总会苦一阵子。只要我们坚持下去，就一定能品到那回甘的滋味，收获美好。"),
        ("广告解说", "https://lf3-static.bytednsdoc.com/obj/eden-cn/lm_hz_ihsph/ljhwZthlaukjlkulzlp/console/bigtts/zh_male_chunhui_mars_bigtts.mp3", "全新升级的这款产品，融合了顶尖科技与时尚设计，它将全方位满足您的需求，成为您生活中不可或缺的好帮手。"),
        ("贴心女声", "https://lf3-static.bytednsdoc.com/obj/eden-cn/lm_hz_ihsph/ljhwZthlaukjlkulzlp/console/bigtts/zh_female_tiexinnvsheng_mars_bigtts.mp3", "音乐是心灵的语言，在音符的跳跃间，能传达出喜怒哀乐。沉浸在音乐的海洋里，仿佛能触摸到灵魂的最深处。"),
        ("俏皮女声", "https://lf3-static.bytednsdoc.com/obj/eden-cn/lm_hz_ihsph/ljhwZthlaukjlkulzlp/console/bigtts/zh_female_qiaopinvsheng_mars_bigtts.mp3", "你这个小迷糊，又忘记东西放在哪里了吧？没关系啦，我来帮你找，下次可不许再这么粗心咯。"),
        ("萌丫头", "https://lf3-static.bytednsdoc.com/obj/eden-cn/lm_hz_ihsph/ljhwZthlaukjlkulzlp/console/bigtts/zh_female_mengyatou_mars_bigtts.mp3", "我今天在花园里看到好多漂亮的蝴蝶，它们飞来飞去像在跳舞，我要是也能像它们一样自由自在就好了。"),
        ("懒音绵宝", "https://lf3-static.bytednsdoc.com/obj/eden-cn/lm_hz_ihsph/ljhwZthlaukjlkulzlp/console/bigtts/zh_male_lanxiaoyang_mars_bigtts.mp3", "哎呀，先睡会儿，有事儿等我睡醒再说吧"),
        ("亮嗓萌仔", "https://lf3-static.bytednsdoc.com/obj/eden-cn/lm_hz_ihsph/ljhwZthlaukjlkulzlp/console/bigtts/zh_male_dongmanhaimian_mars_bigtts.mp3", "比奇堡的海绵，以纯真之心，在奇幻海洋制造欢乐"),
    ],
    "多语种": [
        ("Harmony", "https://lf3-static.bytednsdoc.com/obj/eden-cn/lm_hz_ihsph/ljhwZthlaukjlkulzlp/portal/bigtts/Harmony.mp3", "Hey there! I'm a sports-loving boy! You know, I'm always full of energy and passion for sports. Whether it's running on the track, shooting hoops on the basketball court, or kicking the ball on the soccer field, I'm always in the game."),
        ("Skye", "https://lf3-static.bytednsdoc.com/obj/eden-cn/lm_hz_ihsph/ljhwZthlaukjlkulzlp/portal/bigtts/Skye.mp3", "Hey everyone! I'm a girl who really loves to do meditation. When I sit down cross-legged, close my eyes and focus on my breath, it's like entering a peaceful world of my own. I feel my body and mind gradually relaxing, all the stress and worries seem to fade away."),
        ("Alvin", "https://lf3-static.bytednsdoc.com/obj/eden-cn/lm_hz_ihsph/ljhwZthlaukjlkulzlp/portal/bigtts/Alvin.mp3", "So, what do you want to talk about? Sports? Movies? Music? Or anything else that comes to mind. Let's have a great conversation!"),
        ("Brayan", "https://lf3-static.bytednsdoc.com/obj/eden-cn/lm_hz_ihsph/ljhwZthlaukjlkulzlp/portal/bigtts/Brayan.mp3", "How are you today? I had a really cool day at school today. We had a great science class and I learned some fascinating stuff. And then I played basketball with my friends during the break, it was so much fun. What about you?"),
        ("Adam", "https://lf3-static.bytednsdoc.com/obj/eden-cn/lm_hz_ihsph/ljhwZthlaukjlkulzlp/console/bigtts/en_male_adam_mars_bigtts.mp3", "I'm not afraid of challenges. I believe that with my determination and efforts, I can break through any difficulties and reach the peak of success. I'm ready to face whatever comes my way."),
        ("Shiny", "https://lf3-static.bytednsdoc.com/obj/eden-cn/lm_hz_ihsph/ljhwZthlaukjlkulzlp/console/bigtts/shiny.mp3", "Hey, you seem a bit down. Let's look on the bright side! There's always something good around the corner. A smile can change the world, so let's shine together and make it a better place."),
        ("Morgan", "https://lf3-static.bytednsdoc.com/obj/eden-cn/lm_hz_ihsph/ljhwZthlaukjlkulzlp/console/bigtts/Morgan.mp3", "Listen, I just watched an amazing documentary. The way they presented the story was so captivating. It made me realize how powerful a good narration can be. It can really bring a whole new world to life."),
        ("Hope", "https://lf3-static.bytednsdoc.com/obj/eden-cn/lm_hz_ihsph/ljhwZthlaukjlkulzlp/console/bigtts/Hope.mp3", "You know, even when things seem tough, don't give up. Every setback is a chance to grow. There's always a silver lining. Believe in yourself and keep moving forward. You've got this!"),
        ("Candy", "https://lf3-static.bytednsdoc.com/obj/eden-cn/lm_hz_ihsph/ljhwZthlaukjlkulzlp/console/bigtts/Candy.mp3", "Darling, I noticed you're a bit stressed. How about we relax? We can have a cup of tea, listen to some soft music, and just forget about the worries for a while. You deserve a break."),
        ("Cutey", "https://lf3-static.bytednsdoc.com/obj/eden-cn/lm_hz_ihsph/ljhwZthlaukjlkulzlp/console/bigtts/Cutey.mp3", "Oh, I saw a really cute puppy on the street. It was so fluffy and had the sweetest little face. I wanted to take it home right away. Don't you just love cute things?"),
        ("Jackson", "https://lf3-static.bytednsdoc.com/obj/eden-cn/lm_hz_ihsph/ljhwZthlaukjlkulzlp/console/bigtts/en_male_jackson_mars_bigtts.mp3", "Yo squad turn it UP! Glow sticks in the air. I wanna see mosh pits forming NOW!"),
        ("Amanda", "https://lf3-static.bytednsdoc.com/obj/eden-cn/lm_hz_ihsph/ljhwZthlaukjlkulzlp/console/bigtts/en_female_amanda_mars_bigtts.mp3", "Page 12 foreshadowing the icepick murder – hear that bone-chilling wind through lace curtains?"),
        ("Smith", "https://lf3-static.bytednsdoc.com/obj/eden-cn/lm_hz_ihsph/ljhwZthlaukjlkulzlp/console/bigtts/en_male_smith_mars_bigtts.mp3", "I'm a man of principles. I will never compromise my beliefs for temporary gains. Justice and fairness are what I pursue. I will stand up and fight for what is right, no matter how strong the opposition is."),
        ("Anna", "https://lf3-static.bytednsdoc.com/obj/eden-cn/lm_hz_ihsph/ljhwZthlaukjlkulzlp/console/bigtts/en_female_anna_mars_bigtts.mp3", "Dreams are the stars that light up my path. I won't let obstacles dim their shine. I'll work hard, step by step, to turn those dreams into reality. Because in the pursuit, I find the true meaning and joy of living."),
    ],
    "角色扮演": [
        ("高冷御姐", "https://lf3-static.bytednsdoc.com/obj/eden-cn/lm_hz_ihsph/ljhwZthlaukjlkulzlp/portal/bigtts/高冷御姐.mp3", "哼，亲爱的，我们到此为止吧，我倦了，不想再继续这场无聊的游戏了。你我之间，也许曾经有一些美好，但那也只是曾经罢了。"),
        ("傲娇霸总", "https://lf3-static.bytednsdoc.com/obj/eden-cn/lm_hz_ihsph/ljhwZthlaukjlkulzlp/portal/bigtts/傲娇霸总.mp3", "宝贝，你给我记住，你是我的人，这辈子都是。我不允许你看别的男人一眼，你的心里只能有我。不管你遇到什么事，都有我在，我会为你摆平一切。别总想着离开我，你逃不出我的手掌心的。"),
        ("魅力女友", "https://lf3-static.bytednsdoc.com/obj/eden-cn/lm_hz_ihsph/ljhwZthlaukjlkulzlp/portal/bigtts/魅力女友.mp3", "以后呢，你只能对我一个人好，心里也只能装着我。不管发生什么，都要第一时间想到我哦。我可会一直赖着你的，你别想跑掉啦。还有呀，要好好爱我，宠我，不然我可不依呢，哼！"),
        ("深夜播客", "https://lf3-static.bytednsdoc.com/obj/eden-cn/lm_hz_ihsph/ljhwZthlaukjlkulzlp/portal/bigtts/深夜播客.mp3", "在这寂静的夜里，我陪着你们，一起度过这独特的时光。我知道你们可能带着一天的疲惫来到这里，或是有着各种各样的心情，别担心，有我在呢。让我的声音陪伴你们，为你们赶走孤单，带来一些慰藉和欢乐。"),
        ("柔美女友", "https://lf3-static.bytednsdoc.com/obj/eden-cn/lm_hz_ihsph/ljhwZthlaukjlkulzlp/portal/bigtts/柔美女友.mp3", "亲爱的，这么晚啦，你还没睡呀。我想跟你说哦，不管什么时候，我都会在你身边的呀。你累的时候就靠靠我，不开心了我就哄你开心。"),
        ("撒娇学妹", "https://lf3-static.bytednsdoc.com/obj/eden-cn/lm_hz_ihsph/ljhwZthlaukjlkulzlp/portal/bigtts/撒娇学妹.mp3", "等会儿你一定要好好地、用心地让人家品尝一下你亲自做的美食哟~人家可期待了呢，你做的肯定超级超级好吃，人家现在就已经迫不及待啦，亲爱的最好了啦~"),
        ("病弱少女", "https://lf3-static.bytednsdoc.com/obj/eden-cn/lm_hz_ihsph/ljhwZthlaukjlkulzlp/portal/bigtts/病弱少女.mp3", "你好呀，我感觉今天身体好了一点点，虽然还是有一点虚弱。希望我能赶快好起来，和你一起度过更多美好的时光，谢谢你的关心和陪伴。"),
        ("活泼女孩", "https://lf3-static.bytednsdoc.com/obj/eden-cn/lm_hz_ihsph/ljhwZthlaukjlkulzlp/portal/bigtts/活泼女孩.mp3", "嗨，亲爱的，今天我去一个超级有趣的地方，那里有好多好玩东西，我尝试了一些新的美食，还拍了好多好看的照片呢。"),
        ("东方浩然", "https://lf3-static.bytednsdoc.com/obj/eden-cn/lm_hz_ihsph/ljhwZthlaukjlkulzlp/portal/bigtts/东方浩然.mp3", "我上知天文，从遥远星系的诞生到天体的运行规律，皆能娓娓道来。下知地理，无论是古老文明的发祥地，还是新兴城市的崛起之地，我都能洞察其背后的历史脉络与发展轨迹。"),
        ("奶气萌娃", "https://lf3-static.bytednsdoc.com/obj/eden-cn/lm_hz_ihsph/ljhwZthlaukjlkulzlp/console/bigtts/zh_male_naiqimengwa_mars_bigtts.mp3", "我不要睡觉，星星还没和我说完悄悄话呢，它们会告诉我好多小秘密，等我听完再睡好不好呀？"),
        ("婆婆", "https://lf3-static.bytednsdoc.com/obj/eden-cn/lm_hz_ihsph/ljhwZthlaukjlkulzlp/console/bigtts/zh_female_popo_mars_bigtts.mp3", "你们年轻人啊，做事要沉稳，别毛毛躁躁的，吃亏是福，要懂得包容和体谅，这样才能把日子过得顺顺当当。"),
        ("绿茶小哥", "https://lf3-static.bytednsdoc.com/obj/eden-cn/lm_hz_ihsph/ljhwZthlaukjlkulzlp/console/bigtts/ICL_zh_male_lvchaxiaoge_tob.mp3", "这事儿啊，我本不想多言，但看你如此为难，我就冒险试试吧，只希望别得罪了旁人，我也是一片好心呐。"),
        ("娇弱萝莉", "https://lf3-static.bytednsdoc.com/obj/eden-cn/lm_hz_ihsph/ljhwZthlaukjlkulzlp/console/bigtts/ICL_zh_female_jiaoruoluoli_tob.mp3", "哎呀，人家好怕怕，这个该怎么办呀？你能不能帮帮我，好不好嘛。"),
        ("冷淡疏离", "https://lf3-static.bytednsdoc.com/obj/eden-cn/lm_hz_ihsph/ljhwZthlaukjlkulzlp/console/bigtts/ICL_zh_male_lengdanshuli_tob.mp3", "与我无关之事，莫要打扰。我独自行于这世间，无需他人过多介入。"),
        ("憨厚敦实", "https://lf3-static.bytednsdoc.com/obj/eden-cn/lm_hz_ihsph/ljhwZthlaukjlkulzlp/console/bigtts/ICL_zh_male_hanhoudunshi_tob.mp3", "俺没啥心眼，就知道实实在在做事，你说咋干，俺就咋干，绝不含糊。"),
        ("傲气凌人", "https://lf3-static.bytednsdoc.com/obj/eden-cn/lm_hz_ihsph/ljhwZthlaukjlkulzlp/console/bigtts/ICL_zh_male_aiqilingren_tob.mp3", "哼，这等小事，我轻易便能做到，你们且学着吧，莫要在我面前班门弄斧。"),
        ("活泼刁蛮", "https://lf3-static.bytednsdoc.com/obj/eden-cn/lm_hz_ihsph/ljhwZthlaukjlkulzlp/console/bigtts/ICL_zh_female_huopodiaoman_tob.mp3", "我就要这个，你不给我，我就一直缠着你，哼，看你能拿我怎样。"),
        ("固执病娇", "https://lf3-static.bytednsdoc.com/obj/eden-cn/lm_hz_ihsph/ljhwZthlaukjlkulzlp/console/bigtts/ICL_zh_male_guzhibingjiao_tob.mp3", "你只能属于我，谁也别想靠近，若是违背，我定不会善罢甘休。"),
        ("撒娇粘人", "https://lf3-static.bytednsdoc.com/obj/eden-cn/lm_hz_ihsph/ljhwZthlaukjlkulzlp/console/bigtts/ICL_zh_male_sajiaonianren_tob.mp3", "亲爱的，你不要走嘛，你不在我身边，我心里空落落的，陪陪我好不好。"),
        ("傲慢娇声", "https://lf3-static.bytednsdoc.com/obj/eden-cn/lm_hz_ihsph/ljhwZthlaukjlkulzlp/console/bigtts/ICL_zh_female_aomanjiaosheng_tob.mp3", "你们这些人，都要好好伺候本小姐，若是有差池，定不轻饶。"),
        ("潇洒随性", "https://lf3-static.bytednsdoc.com/obj/eden-cn/lm_hz_ihsph/ljhwZthlaukjlkulzlp/console/bigtts/ICL_zh_male_xiaosasuixing_tob.mp3", "世间纷扰，何必拘泥，随心而为，方得自在，走，一起去看那未知风景。"),
    ],
    "通用场景": [
        ("爽快思思", "https://lf3-static.bytednsdoc.com/obj/eden-cn/lm_hz_ihsph/ljhwZthlaukjlkulzlp/portal/bigtts/爽快思思.mp3", "今天天气可好了，我打算和朋友一起去野餐，带上美食和饮料，找个舒适的草坪，什么烦恼都没了。你要不要和我们一起呀？"),
        ("温暖阿虎", "https://lf3-static.bytednsdoc.com/obj/eden-cn/lm_hz_ihsph/ljhwZthlaukjlkulzlp/portal/bigtts/温暖阿虎.mp3", "早上好呀，今天的阳光特别灿烂，就像你的笑容一样，让我倍感温暖。"),
        ("邻家女孩", "https://lf3-static.bytednsdoc.com/obj/eden-cn/lm_hz_ihsph/ljhwZthlaukjlkulzlp/portal/bigtts/邻家女孩.mp3", "哎呀，你来找我啦！今天过得怎么样呀？我今天看到院子里的花开了呢，可漂亮啦！你想不想和我一起去看看呀？"),
        ("少年梓辛", "https://lf3-static.bytednsdoc.com/obj/eden-cn/lm_hz_ihsph/ljhwZthlaukjlkulzlp/portal/bigtts/少年梓辛.mp3", "今天的阳光真好啊！感觉整个人都充满了活力呢。真想去外面跑一跑，去探索那些有趣的地方，去邂逅一些奇妙的事情。生活嘛，就该这样自由自在、充满朝气的呀！哈哈！"),
        ("渊博小叔", "https://lf3-static.bytednsdoc.com/obj/eden-cn/lm_hz_ihsph/ljhwZthlaukjlkulzlp/portal/bigtts/渊博小叔.mp3", "你要知道，这世间的知识就如同浩瀚的海洋，无穷无尽啊。我们要始终保持一颗求知的心，不断去探索、去学习。无论是科学、历史、文化还是艺术，每一个领域都有着无尽的奥秘等待我们去揭开。"),
        ("阳光青年", "https://lf3-static.bytednsdoc.com/obj/eden-cn/lm_hz_ihsph/ljhwZthlaukjlkulzlp/portal/bigtts/阳光青年.mp3", "今天又是超棒的一天呀！阳光这么好，心情也跟着超级美丽呢！生活嘛，就该充满活力和欢笑呀！我呀，要像那灿烂的阳光一样，永远积极向上，去追寻自己的梦想，去体验各种好玩的事情，去认识更多有趣的人！"),
        ("甜美小源", "https://lf3-static.bytednsdoc.com/obj/eden-cn/lm_hz_ihsph/ljhwZthlaukjlkulzlp/portal/bigtts/甜美小源.mp3", "你好，我是你的虚拟助理，我随时在这里，陪你聊聊天，分享生活中的喜怒哀乐哦。如果你有任何问题或者需要建议，都可以随时问我呢。期待你的分享。"),
        ("清澈梓梓", "https://lf3-static.bytednsdoc.com/obj/eden-cn/lm_hz_ihsph/ljhwZthlaukjlkulzlp/portal/bigtts/清澈梓梓.mp3", "你好呀！我最近对瑜伽特 别着迷，每当我铺开瑜伽垫，舒展身体，仿佛进入了一个宁静而美好的世界。"),
        ("解说小明", "https://lf3-static.bytednsdoc.com/obj/eden-cn/lm_hz_ihsph/ljhwZthlaukjlkulzlp/portal/bigtts/解说小明.mp3", "嘿，你好呀！今天的阳光格外温暖，就像你的笑容一样，瞬间照亮了我的世界。刚刚看到你的那一刻，我就觉得有一种特别的吸引力。"),
        ("开朗姐姐", "https://lf3-static.bytednsdoc.com/obj/eden-cn/lm_hz_ihsph/ljhwZthlaukjlkulzlp/portal/bigtts/开朗姐姐.mp3", "嘿，你好！在这个丰富多彩的世界里，有无数的风景等待着我们去探索。不知道你是否也和我一样，对远方充满了好奇与向往呢？"),
        ("邻家男孩", "https://lf3-static.bytednsdoc.com/obj/eden-cn/lm_hz_ihsph/ljhwZthlaukjlkulzlp/portal/bigtts/邻家男孩.mp3", "今天在这里遇到你，真的感觉特别奇妙。就好像命运的齿轮悄然转动，让我们在这个特定的时间和地点相遇。这一定是一种特别的缘分吧。你看，周围的人来人往，而我的目光却不由自主地被你吸引。"),
        ("甜美悦悦", "https://lf3-static.bytednsdoc.com/obj/eden-cn/lm_hz_ihsph/ljhwZthlaukjlkulzlp/portal/bigtts/甜美悦悦.mp3", "你喜欢看电影吗？电影就像是一个个奇妙的世界，能让我们沉浸其中，体验各种不同的人生。有搞笑的喜剧让我们开怀大笑，有感人的剧情片让我们热泪盈眶，你最喜欢哪种类型的电影呢？"),
        ("心灵鸡汤", "https://lf3-static.bytednsdoc.com/obj/eden-cn/lm_hz_ihsph/ljhwZthlaukjlkulzlp/portal/bigtts/心灵鸡汤.mp3", "人生的意义是不断的追求。不要等错过了才悔恨，不要等老了才怀念。抓住当下，再苦再累也要展翅飞翔。"),
        ("灿灿", "https://lf3-static.bytednsdoc.com/obj/eden-cn/lm_hz_ihsph/ljhwZthlaukjlkulzlp/console/bigtts/zh_female_cancan_mars_bigtts.mp3", "刚刚还在想你怎么还不来找我聊天，你就来了，真是心有灵犀呀。"),
        ("知性女声", "https://lf3-static.bytednsdoc.com/obj/eden-cn/lm_hz_ihsph/ljhwZthlaukjlkulzlp/console/bigtts/zh_female_zhixingnvsheng_mars_bigtts.mp3", "在文学的世界里漫步，如同与无数智者倾心交谈，从诗词的优美到散文的灵动，每一种文字都能让心灵得到滋养与慰藉。"),
        ("清新女声", "https://lf3-static.bytednsdoc.com/obj/eden-cn/lm_hz_ihsph/ljhwZthlaukjlkulzlp/console/bigtts/zh_female_qingxinnvsheng_mars_bigtts.mp3", "清晨的第一缕阳光洒下，仿佛为世界披上了一层金色的纱衣，在这宁静美好的时刻，感受着生命的温柔与力量。"),
        ("清爽男大", "https://lf3-static.bytednsdoc.com/obj/eden-cn/lm_hz_ihsph/ljhwZthlaukjlkulzlp/console/bigtts/zh_male_qingshuangnanda_mars_bigtts.mp3", "青春就是要敢闯敢拼，不怕失败。我们要在这热血的年纪，去追逐梦想，去体验不同的风景，让青春不留遗憾。"),
        ("知性温婉", "https://lf3-static.bytednsdoc.com/obj/eden-cn/lm_hz_ihsph/ljhwZthlaukjlkulzlp/console/bigtts/ICL_zh_female_zhixingwenwan_tob.mp3", "生活的美好常隐匿于细微之处，我们需以平和之心去感知，方能领略其真谛，愿你也能有此心境。"),
        ("暖心体贴", "https://lf3-static.bytednsdoc.com/obj/eden-cn/lm_hz_ihsph/ljhwZthlaukjlkulzlp/console/bigtts/ICL_zh_male_nuanxintitie_tob.mp3", "你看起来有些疲惫，是不是累了？先休息一下吧，我帮你把事情处理好。"),
        ("温柔文雅", "https://lf3-static.bytednsdoc.com/obj/eden-cn/lm_hz_ihsph/ljhwZthlaukjlkulzlp/console/bigtts/ICL_zh_female_wenrouwenya_tob.mp3", "清风徐来，水波不兴，世间纷扰，亦当以优雅之态处之，心平气和，方显从容。"),
        ("开朗轻快", "https://lf3-static.bytednsdoc.com/obj/eden-cn/lm_hz_ihsph/ljhwZthlaukjlkulzlp/console/bigtts/ICL_zh_male_kailangqingkuai_tob.mp3", "今天又是超棒的一天，不管遇到啥，都不能影响我的好心情，冲呀！"),
    ],
}

def download_file(url: str, save_path: Path) -> bool:
    """下载文件"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            f.write(response.content)
        return True
    except Exception as e:
        print(f"  ✗ 下载失败: {e}")
        return False

def download_voices(category: str, voices: list):
    """下载指定分类的音频"""
    category_dir = BASE_DIR / category
    category_dir.mkdir(parents=True, exist_ok=True)

    for voice_name, mp3_url, text in voices:
        safe_name = voice_name.replace('/', '_').replace('\\', '_')
        mp3_path = category_dir / f"{safe_name}.mp3"
        txt_path = category_dir / f"{safe_name}.txt"

        # 下载 MP3
        if not mp3_path.exists():
            print(f"下载 {category}/{safe_name}.mp3 ...")
            if download_file(mp3_url, mp3_path):
                print(f"  ✓ 成功")
            else:
                continue
        else:
            print(f"  跳过已存在: {category}/{safe_name}.mp3")

        # 写入文本文件
        if text and not txt_path.exists():
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"  ✓ 创建文本文件")
        elif text and txt_path.exists():
            # 更新文本文件
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"  ✓ 更新文本文件")

        time.sleep(0.3)

def main():
    print("=" * 60)
    print("补充下载豆包语音合成预览音频")
    print("=" * 60)

    total = sum(len(v) for v in COMPLETE_VOICE_DATA.values())
    print(f"总计 {len(COMPLETE_VOICE_DATA)} 个分类, {total} 个音色\n")

    for category, voices in COMPLETE_VOICE_DATA.items():
        print(f"\n=== {category} ({len(voices)} 个音色) ===")
        download_voices(category, voices)

    print("\n" + "=" * 60)
    print("补充下载完成!")
    print("=" * 60)

if __name__ == "__main__":
    main()
