# was last updated on 23-7-2025
# rewrited  hybrid model
# this code was made by Vihaan Kanwar
# last attempt to fix the model was on 22-7-2025 ( reason was high inaccuracy)
# was tested on 23-7-2025

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    pipeline
)
from torch.amp import autocast
import numpy as np
import logging
from typing import Dict, Tuple, List
import re
from sentence_transformers import SentenceTransformer
import os
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridEmotionClassifier(nn.Module):
    def __init__(self, 
                 primary_model='roberta-large',
                 secondary_model='all-MiniLM-L6-v2',
                 num_labels=6, 
                 dropout_rate=0.3,
                 lstm_hidden=512,
                 attention_heads=8):
        super().__init__()
        
        
        self.config = AutoConfig.from_pretrained(primary_model)
        self.transformer = AutoModel.from_pretrained(
            primary_model, 
            output_attentions=True,
            attn_implementation="eager"  
        )
        hidden_size = self.config.hidden_size
        
        
        
        for i, layer in enumerate(self.transformer.encoder.layer):
            if i < 12:  
                for param in layer.parameters():
                    param.requires_grad_(False)
            else:  
                for param in layer.parameters():
                    param.requires_grad_(True)
        
        
        for param in self.transformer.embeddings.parameters():
            param.requires_grad_(False)
        
        self.lstm = nn.LSTM(
            hidden_size, lstm_hidden, 
            batch_first=True, bidirectional=True, 
            num_layers=2, dropout=dropout_rate  
        )
        
        self.attention = nn.MultiheadAttention(
            lstm_hidden * 2, num_heads=attention_heads, 
            dropout=dropout_rate, batch_first=True
        )
        
        self.context_attention = nn.Linear(lstm_hidden * 2, 1)
        
        
        fusion_dim = hidden_size + lstm_hidden * 2 + 384
        self.feature_fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),  
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.BatchNorm1d(hidden_size // 4),
            nn.GELU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(hidden_size // 4, num_labels)
        )
        
        self.confidence_estimator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.2),  
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in [self.feature_fusion, self.classifier, self.confidence_estimator]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, input_ids, attention_mask, sentence_embeddings=None):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        lstm_out, (h_n, c_n) = self.lstm(hidden_states)
        lstm_out = self.dropout(lstm_out)
        
        attn_output, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        context_weights = F.softmax(self.context_attention(attn_output), dim=1)
        lstm_pooled = torch.sum(attn_output * context_weights, dim=1)
        
        cls_output = hidden_states[:, 0, :]
        
        if sentence_embeddings is not None:
            combined = torch.cat([cls_output, lstm_pooled, sentence_embeddings], dim=1)
        else:
            combined = torch.cat([cls_output, lstm_pooled, torch.zeros(cls_output.size(0), 384).to(cls_output.device)], dim=1)
        
        fused_features = self.feature_fusion(combined)
        
        logits = self.classifier(fused_features)
        
        confidence_score = self.confidence_estimator(fused_features)
        
        return logits, confidence_score, attn_weights

class ImprovedEmotionPreprocessor:
    def __init__(self):
        
        self.emotion_patterns = {
            'sadness': [
                r'\b(sad|depressed|down|blue|melancholy|gloomy|miserable|unhappy|sorrowful|dejected)\b',
                r'\b(cry|tears|weep|sob|crying|tearful)\b',
                r'\b(hopeless|despair|grief|sorrow|disappointed|heartbroken|devastated)\b',
                r'\b(nothing|wrong|feel|bad|worse|terrible|awful)\b'
            ],
            'joy': [
                r'\b(happy|joy|glad|cheerful|elated|ecstatic|thrilled|excited|delighted)\b',
                r'\b(laugh|smile|grin|giggle|laughing|smiling)\b',
                r'\b(awesome|amazing|wonderful|fantastic|great|excellent|perfect|brilliant)\b',
                r'\b(celebrating|celebration|party|fun|enjoy)\b'
            ],
            'anger': [
                r'\b(angry|mad|furious|rage|irritated|annoyed|pissed|outraged)\b',
                r'\b(hate|disgusted|upset|frustrated|aggravated|infuriated)\b',
                r'\b(damn|hell|stupid|idiot|ridiculous|absurd)\b',
                r'\b(betrayed|betrayal|unfair|injustice)\b'
            ],
            'fear': [
                r'\b(scared|afraid|terrified|frightened|anxious|worried|nervous)\b',
                r'\b(panic|dread|uneasy|concerned|apprehensive|alarmed)\b',
                r'\b(threat|danger|risk|unsafe|insecure)\b'
            ],
            'love': [
                r'\b(love|adore|cherish|treasure|affection|beloved|dear)\b',
                r'\b(romantic|intimate|caring|tender|sweet|darling)\b',
                r'\b(family|relationship|partner|husband|wife|girlfriend|boyfriend)\b'
            ],
            'surprise': [
                r'\b(surprised|shocked|amazed|astonished|stunned|bewildered)\b',
                r'\b(wow|omg|incredible|unbelievable|unexpected|sudden)\b',
                r'\b(can\'t believe|never expected|out of nowhere)\b'
            ]
        }
        
        
        self.negation_patterns = [
            r'\b(not|never|no|don\'t|doesn\'t|didn\'t|won\'t|wouldn\'t|can\'t|couldn\'t)\b'
        ]
    
    def extract_emotion_features(self, text: str) -> Dict[str, float]:
        text_lower = text.lower()
        features = {}
        
        
        negation_count = sum(len(re.findall(pattern, text_lower)) for pattern in self.negation_patterns)
        has_negation = negation_count > 0
        
        for emotion, patterns in self.emotion_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower))
                score += matches
            
            
            if has_negation and emotion in ['joy', 'love']:
                score *= 0.3  
            elif has_negation and emotion in ['sadness', 'anger', 'fear']:
                score *= 1.2  
                
            features[f'{emotion}_keywords'] = score
        
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['caps_ratio'] = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        features['negation_count'] = negation_count
        
        return features

class EmotionClassifier:
    def __init__(self, 
                 model_name='roberta-large',
                 sentence_model='all-MiniLM-L6-v2',
                 checkpoint_path=None,
                 device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.sentence_transformer = SentenceTransformer(sentence_model)
        
        self.model = HybridEmotionClassifier(
            primary_model=model_name,
            secondary_model=sentence_model,
            num_labels=6,
            dropout_rate=0.3,
            lstm_hidden=512,
            attention_heads=8
        )
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                if 'model_state_dict' in checkpoint:
                    
                    missing_keys, unexpected_keys = self.model.load_state_dict(
                        checkpoint['model_state_dict'], strict=False
                    )
                    if missing_keys:
                        logger.warning(f"Missing keys: {missing_keys}")
                    if unexpected_keys:
                        logger.warning(f"Unexpected keys: {unexpected_keys}")
                else:
                    self.model.load_state_dict(checkpoint, strict=False)
                logger.info("Checkpoint loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load checkpoint: {e}")
                logger.info("Using randomly initialized model")
        else:
            logger.info("No checkpoint provided or file not found, using randomly initialized model")
        
        self.model.to(self.device)
        self.model.eval()
        
        
        try:
            self.llm_pipeline = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                device=0 if torch.cuda.is_available() else -1,
                return_all_scores=True  # Get all emotion scores
            )
        except Exception as e:
            logger.warning(f"Could not initialize LLM pipeline: {e}")
            self.llm_pipeline = None
        
        self.preprocessor = ImprovedEmotionPreprocessor()
        
        self.emotion_map = {
            0: 'Sadness', 1: 'Joy', 2: 'Love', 
            3: 'Anger', 4: 'Fear', 5: 'Surprise'
        }
        
        
        self.llm_to_model_map = {
            'sadness': 0, 'joy': 1, 'love': 2,
            'anger': 3, 'fear': 4, 'surprise': 5
        }
    
    def _get_sentence_embeddings(self, texts: List[str]) -> torch.Tensor:
        embeddings = self.sentence_transformer.encode(texts, convert_to_tensor=True)
        return embeddings.to(self.device)
    
    def _get_llm_prediction(self, text: str) -> Tuple[Dict[str, float], str, float]:
        if self.llm_pipeline is None:
            return {}, None, 0.0
        
        try:
            results = self.llm_pipeline(text)
            if results and len(results) > 0:
                
                emotion_scores = {result['label'].lower(): result['score'] for result in results[0]}
                
                
                top_result = max(results[0], key=lambda x: x['score'])
                emotion = top_result['label'].lower()
                confidence = top_result['score']
                
                return emotion_scores, emotion, confidence
        except Exception as e:
            logger.warning(f"LLM prediction failed: {e}")
        
        return {}, None, 0.0
    
    def predict_emotion(self, text: str, use_ensemble=True) -> Dict:
        text = text.strip()
        if not text:
            return {
                'emotion': 'Unknown',
                'confidence': 0.0,
                'probabilities': [0.0] * 6,
                'method': 'empty_input'
            }
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=512,  
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        sentence_embeddings = self._get_sentence_embeddings([text])
        
        with torch.no_grad():
            with autocast('cuda', enabled=self.device.type == 'cuda'):
                logits, confidence_score, attention_weights = self.model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask,
                    sentence_embeddings=sentence_embeddings
                )
                
                probabilities = F.softmax(logits, dim=1).cpu().numpy()[0]
                main_prediction = int(torch.argmax(logits, dim=1).cpu().numpy()[0])
                main_confidence = float(confidence_score.cpu().numpy().item())
        
        if use_ensemble:
            
            emotion_scores, llm_emotion, llm_confidence = self._get_llm_prediction(text)
            emotion_features = self.preprocessor.extract_emotion_features(text)
            
            final_probabilities = probabilities.copy()
            final_confidence = main_confidence
            
            
            keyword_boost = 0.0
            for emotion in self.emotion_map.values():
                emotion_key = f'{emotion.lower()}_keywords'
                if emotion_key in emotion_features and emotion_features[emotion_key] > 0:
                    emotion_idx = [k for k, v in self.emotion_map.items() if v == emotion][0]
                    boost_factor = min(0.3, emotion_features[emotion_key] * 0.1)
                    final_probabilities[emotion_idx] += boost_factor
                    keyword_boost = max(keyword_boost, boost_factor)
            
            
            final_probabilities = final_probabilities / np.sum(final_probabilities)
            
            
            if llm_emotion and llm_emotion in self.llm_to_model_map and llm_confidence > 0.5:
                llm_pred_idx = self.llm_to_model_map[llm_emotion]
                
                
                weight = 0.3 if llm_confidence > 0.8 else 0.2
                final_probabilities = (1 - weight) * final_probabilities + weight * np.array([
                    llm_confidence if i == llm_pred_idx else (1 - llm_confidence) / 5 
                    for i in range(6)
                ])
                
                final_confidence = max(main_confidence, (main_confidence + llm_confidence) / 2)
            
            
            if keyword_boost > 0:
                final_confidence = min(0.95, final_confidence + keyword_boost)
            
            final_prediction = np.argmax(final_probabilities)
            method = 'ensemble'
        else:
            final_probabilities = probabilities
            final_prediction = main_prediction
            final_confidence = main_confidence
            method = 'main_model'
        
        
        text_length = len(text.split())
        if text_length < 3:
            final_confidence *= 0.7
        elif text_length > 100:
            final_confidence *= 0.85
        
        
        final_confidence = max(0.1, min(0.99, final_confidence))
        
        return {
            'emotion': self.emotion_map[final_prediction],
            'confidence': float(final_confidence),
            'probabilities': final_probabilities.tolist(),
            'raw_probabilities': probabilities.tolist(),
            'attention_weights': attention_weights.cpu().numpy() if attention_weights is not None else None,
            'method': method,
            'text_features': emotion_features if use_ensemble else None,
            'llm_scores': emotion_scores if use_ensemble else None
        }
    
    def batch_predict(self, texts: List[str], batch_size: int = 8) -> List[Dict]:
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_results = [self.predict_emotion(text) for text in batch_texts]
            results.extend(batch_results)
        
        return results
    
    def save(self, save_path: str, include_tokenizer: bool = True, metadata: Dict = None):
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        
        model_state = {
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'primary_model': 'roberta-large',
                'secondary_model': 'all-MiniLM-L6-v2',
                'num_labels': 6,
                'dropout_rate': 0.3,
                'lstm_hidden': 512,
                'attention_heads': 8
            },
            'emotion_map': self.emotion_map,
            'llm_to_model_map': self.llm_to_model_map,
            'save_timestamp': datetime.now().isoformat(),
            'device': str(self.device),
            'metadata': metadata or {}
        }
        
        torch.save(model_state, f"{save_path}.pth")
        logger.info(f"Model saved to {save_path}.pth")
        
        if include_tokenizer:
            self.tokenizer.save_pretrained(f"{save_path}_tokenizer")
            logger.info(f"Tokenizer saved to {save_path}_tokenizer/")
        
        print(f"Model saved successfully to {save_path}.pth")
    
    @classmethod
    def load(cls, model_path: str, device=None):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        checkpoint = torch.load(model_path, map_location=device)
        
        config = checkpoint.get('model_config', {})
        
        classifier = cls(
            model_name=config.get('primary_model', 'roberta-large'),
            sentence_model=config.get('secondary_model', 'all-MiniLM-L6-v2'),
            device=device
        )
        
        classifier.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        if 'emotion_map' in checkpoint:
            classifier.emotion_map = checkpoint['emotion_map']
        if 'llm_to_model_map' in checkpoint:
            classifier.llm_to_model_map = checkpoint['llm_to_model_map']
        
        classifier.model.eval()
        
        logger.info(f"Model loaded successfully from {model_path}")
        logger.info(f"Device: {device}")
        logger.info(f"Emotion classes: {list(classifier.emotion_map.values())}")
        
        return classifier

def main():
    
    classifier = EmotionClassifier(
        model_name='roberta-large',
        checkpoint_path=None  
    )
    
    test_texts = [
        "I don't know what's wrong with me anymore. Nothing feels good lately.",
        "I'm so happy today! Everything is going perfectly!",
        "I love spending time with my family.",
        "This situation makes me so angry and frustrated!",
        "I'm really scared about the upcoming exam.",
        "Wow, I can't believe this amazing surprise!",
        "I don't know what's wrong with me anymore. Nothing feels good lately. I really think the Jedi Council has betrayed me by not letting me be a master, and torturing me my entire life. I am not the jedi i used to be."
    ]
    
    print("=== Fixed Hybrid Emotion Classifier Results ===\n")
    
    for text in test_texts:
        result = classifier.predict_emotion(text, use_ensemble=True)
        
        print(f"[+] Text: '{text}'")
        print(f"[+] Predicted: {result['emotion']} (confidence: {result['confidence']:.3f})")
        print(f"[+] Method: {result['method']}")
        
        emotion_probs = list(zip(classifier.emotion_map.values(), result['probabilities']))
        emotion_probs.sort(key=lambda x: x[1], reverse=True)
        print("[?] Top 3 predictions:")
        for emotion, prob in emotion_probs[:3]:
            print(f"  {emotion}: {prob:.3f}")
        
        if result['text_features']:
            keyword_features = {k: v for k, v in result['text_features'].items() if '_keywords' in k and v > 0}
            if keyword_features:
                print(f"[+] Detected keywords: {keyword_features}")
        
        if result['llm_scores']:
            print(f"[+] LLM scores: {result['llm_scores']}")
        
        print("-" * 60)
    
    
    save_path = "fixed_emotion_classifier_model"
    classifier.save(save_path, include_tokenizer=True, metadata={"version": "2.0", "author": "vihaan kanwar", "fixes": "attn_implementation, ensemble improvements, better patterns"})
    print(f"[+] Model saved successfully to {save_path}.pth")

if __name__ == "__main__":
    main()