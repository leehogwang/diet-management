import 'dart:convert';
import 'package:http/http.dart' as http;

// 식품 영양 정보 데이터 모델
class FoodNutritionInfo {
  final String foodName;
  final double calories;
  final double carbohydrates;
  final double protein;
  final double fat;
  final String? manufacturer; // 제조사
  final String? servingSize; // 1회 제공량

  FoodNutritionInfo({
    required this.foodName,
    required this.calories,
    required this.carbohydrates,
    required this.protein,
    required this.fat,
    this.manufacturer,
    this.servingSize,
  });

  factory FoodNutritionInfo.fromJson(Map<String, dynamic> json) {
    // 식약처 API 응답 형식에 맞게 파싱
    // 실제 API 응답 구조에 따라 수정 필요
    return FoodNutritionInfo(
      foodName: json['DESC_KOR'] ?? json['FOOD_NM_KR'] ?? '',
      calories: _parseDouble(json['NUTR_CONT1']), // 에너지(kcal)
      carbohydrates: _parseDouble(json['NUTR_CONT2']), // 탄수화물(g)
      protein: _parseDouble(json['NUTR_CONT3']), // 단백질(g)
      fat: _parseDouble(json['NUTR_CONT4']), // 지방(g)
      manufacturer: json['MAKER_NM'],
      servingSize: json['SERVING_SIZE'],
    );
  }

  static double _parseDouble(dynamic value) {
    if (value == null) return 0.0;
    if (value is double) return value;
    if (value is int) return value.toDouble();
    if (value is String) {
      return double.tryParse(value) ?? 0.0;
    }
    return 0.0;
  }
}

class FoodNutritionAPI {
  // API 키는 발급받은 후 여기에 입력하세요
  static String apiKey = '0348c25a8c52f6b3fd54b1e82ab0a22ca7134289610c305438667854c802aac5';

  // 식약처 식품영양성분 DB API 엔드포인트
  static const String baseUrl = 'http://apis.data.go.kr/1471000/FoodNtrIrdntInfoService1';

  // API 키 설정
  static void setApiKey(String key) {
    apiKey = key;
  }

  // 식품 검색 (getFoodNtrItdntList1 - 식품영양성분DB 조회 서비스)
  static Future<List<FoodNutritionInfo>> searchFood(String foodName) async {
    if (apiKey == 'YOUR_API_KEY_HERE') {
      throw Exception('API 키가 설정되지 않았습니다. FoodNutritionAPI.setApiKey()를 먼저 호출하세요.');
    }

    try {
      final uri = Uri.parse('$baseUrl/getFoodNtrItdntList1').replace(
        queryParameters: {
          'serviceKey': apiKey,
          'type': 'json',
          'pageNo': '1',
          'numOfRows': '10',
          'FOOD_NM_KR': foodName, // 식품명으로 검색
        },
      );

      final response = await http.get(uri);

      if (response.statusCode == 200) {
        final data = json.decode(response.body);

        // API 응답 구조 확인 (실제 응답에 따라 수정 필요)
        final items = data['body']?['items'] as List?;

        if (items == null || items.isEmpty) {
          return [];
        }

        return items
            .map((item) => FoodNutritionInfo.fromJson(item))
            .toList();
      } else {
        throw Exception('API 요청 실패: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('식품 정보 검색 중 오류 발생: $e');
    }
  }

  // 식품 코드로 상세 정보 조회
  static Future<FoodNutritionInfo?> getFoodDetailByCode(String foodCode) async {
    if (apiKey == 'YOUR_API_KEY_HERE') {
      throw Exception('API 키가 설정되지 않았습니다.');
    }

    try {
      final uri = Uri.parse('$baseUrl/getFoodNtrItdntList1').replace(
        queryParameters: {
          'serviceKey': apiKey,
          'type': 'json',
          'FOOD_CD': foodCode,
        },
      );

      final response = await http.get(uri);

      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        final items = data['body']?['items'] as List?;

        if (items != null && items.isNotEmpty) {
          return FoodNutritionInfo.fromJson(items.first);
        }
        return null;
      } else {
        throw Exception('API 요청 실패: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('식품 상세 정보 조회 중 오류 발생: $e');
    }
  }

  // API 키 검증
  static Future<bool> validateApiKey() async {
    if (apiKey == 'YOUR_API_KEY_HERE') {
      return false;
    }

    try {
      final uri = Uri.parse('$baseUrl/getFoodNtrItdntList1').replace(
        queryParameters: {
          'serviceKey': apiKey,
          'type': 'json',
          'pageNo': '1',
          'numOfRows': '1',
        },
      );

      final response = await http.get(uri);
      return response.statusCode == 200;
    } catch (e) {
      return false;
    }
  }
}
