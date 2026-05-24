import re
import string

class ChineseTextNormalizer:
    """
    中文文本标准化器，用于CER计算前的文本预处理
    """
    
    def __init__(self):
        # 中文标点符号
        self.chinese_punctuation = "，。！？；：""''（）【】《》〈〉、·…—～"
        # 英文标点符号
        self.english_punctuation = string.punctuation
        # 所有要移除的标点符号
        self.all_punctuation = self.chinese_punctuation + self.english_punctuation
        
        # 数字映射 - 中文数字转阿拉伯数字
        self.chinese_numbers = {
            '零': '0', '一': '1', '二': '2', '三': '3', '四': '4',
            '五': '5', '六': '6', '七': '7', '八': '8', '九': '9',
            '十': '10'
        }
        
    def remove_punctuation(self, text):
        """移除所有标点符号"""
        for punct in self.all_punctuation:
            text = text.replace(punct, '')
        return text
    
    def normalize_spaces(self, text):
        """标准化空格 - 移除多余空格"""
        # 移除所有空格（中文通常不需要空格）
        text = re.sub(r'\s+', '', text)
        return text
    
    def normalize_numbers(self, text):
        """标准化数字 - 可选的中文数字转换"""
        for chinese_num, arabic_num in self.chinese_numbers.items():
            text = text.replace(chinese_num, arabic_num)
        return text
    
    def normalize_case(self, text):
        """转换为小写（主要影响英文字符）"""
        return text.lower()
    
    def __call__(self, text):
        """
        对文本进行完整的标准化处理
        
        Args:
            text (str): 输入文本
            
        Returns:
            str: 标准化后的文本
        """
        if not text:
            return ""
            
        # 1. 转换为小写
        text = self.normalize_case(text)
        
        # 2. 移除标点符号
        text = self.remove_punctuation(text)
        
        # 3. 标准化空格
        text = self.normalize_spaces(text)
        
        # 4. 可选：标准化数字
        text = self.normalize_numbers(text)
        
        return text.strip() 