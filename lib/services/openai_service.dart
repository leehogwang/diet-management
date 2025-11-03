import 'dart:io';
import 'dart:convert';
import 'package:http/http.dart' as http;

class ClaudeService {
  final String apiKey;

  ClaudeService(this.apiKey);

  Future<Map<String, dynamic>> analyzeFoodImage(File imageFile) async {
    try {
      print('🔍 Starting food analysis with Claude...');
      print('🔑 API Key: ${apiKey.substring(0, 20)}...');

      // 이미지를 base64로 변환
      final bytes = await imageFile.readAsBytes();
      final base64Image = base64Encode(bytes);
      print('✅ Image converted to base64 (length: ${base64Image.length})');

      print('📤 Sending request to Claude API...');

      // Claude Sonnet 4.5 (2025년 최신 모델 - 코딩 및 에이전트 작업에 최적화)
      final response = await http.post(
        Uri.parse('https://api.anthropic.com/v1/messages'),
        headers: {
          'Content-Type': 'application/json',
          'x-api-key': apiKey,
          'anthropic-version': '2023-06-01',
        },
        body: jsonEncode({
          'model': 'claude-sonnet-4-5-20250929', 
          'max_tokens': 1024,
          'messages': [
            {
              'role': 'user',
              'content': [
                {
                  'type': 'image',
                  'source': {
                    'type': 'base64',
                    'media_type': 'image/jpeg',
                    'data': base64Image,
                  }
                },
                {
                  'type': 'text',
                  'text': '''이 음식 사진을 분석하고 다음 정보를 JSON 형식으로 반환해주세요:

{
  "foods": [
    {
      "name": "음식명",
      "x": 0.5,
      "y": 0.5,
      "nutrition": {
        "calories": 해당_음식_칼로리,
        "sugar": 해당_음식_당_g,
        "protein": 해당_음식_단백질_g,
        "fat": 해당_음식_지방_g,
        "sodium": 해당_음식_나트륨_mg,
        "carbohydrates": 해당_음식_탄수화물_g
      }
    }
  ],
  "nutrition": {
    "calories": 총_칼로리,
    "sugar": 총_당_g,
    "protein": 총_단백질_g,
    "fat": 총_지방_g,
    "sodium": 총_나트륨_mg,
    "carbohydrates": 총_탄수화물_g
  }
}

각 음식의 위치(x, y)는 0.0~1.0 사이 값이며, 각 음식마다 개별 영양 정보를 포함해야 합니다.
JSON만 반환하세요. 다른 설명은 필요없습니다.'''
                }
              ]
            }
          ],
        }),
      ).timeout(
        const Duration(seconds: 60),
        onTimeout: () {
          throw Exception('API 요청 타임아웃 (60초 초과)');
        },
      );

      print('📡 API Response Status: ${response.statusCode}');

      if (response.statusCode == 200) {
        final data = jsonDecode(utf8.decode(response.bodyBytes));
        print('📦 Full API response: $data');

        final content = data['content'] as List?;
        if (content == null || content.isEmpty) {
          throw Exception('API 응답에 content가 없습니다');
        }

        final textContent = content[0]['text'] as String? ?? '';
        if (textContent.isEmpty) {
          throw Exception('API 응답이 비어있습니다');
        }

        print('✅ Response received: ${textContent.substring(0, textContent.length > 100 ? 100 : textContent.length)}...');

        // JSON 파싱
        final jsonMatch = RegExp(r'\{[\s\S]*\}').firstMatch(textContent);
        if (jsonMatch != null) {
          final jsonString = jsonMatch.group(0)!;
          final result = jsonDecode(jsonString) as Map<String, dynamic>;
          print('✅ JSON parsed successfully: $result');
          return result;
        }

        throw Exception('JSON 파싱 실패: $textContent');
      } else {
        final errorBody = utf8.decode(response.bodyBytes);
        print('❌ API Error: $errorBody');
        throw Exception('API 요청 실패 (${response.statusCode}): $errorBody');
      }
    } catch (e, stackTrace) {
      print('❌ Error analyzing food: $e');
      print('📍 Stack trace: $stackTrace');
      rethrow;
    }
  }
}
