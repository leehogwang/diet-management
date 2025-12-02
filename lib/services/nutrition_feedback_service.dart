import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:flutter_dotenv/flutter_dotenv.dart';
import '../models/food_record.dart';

class NutritionFeedbackService {
  static const String apiUrl = 'https://api.anthropic.com/v1/messages';

  static Future<String> generateFeedback(
    List<FoodRecord> records,
    DailyNutritionSummary summary,
  ) async {
    try {
      final apiKey = dotenv.env['ANTHROPIC_API_KEY'];

      if (apiKey == null || apiKey.isEmpty) {
        return _generateRuleBasedFeedback(records, summary);
      }

      // 음식 기록을 텍스트로 변환
      final foodList = records.map((r) {
        return '- ${r.mealType}: ${r.foodName} (칼로리: ${r.calories}kcal, 단백질: ${r.protein}g, 탄수화물: ${r.carbs}g, 지방: ${r.fat}g, 나트륨: ${r.sodium}mg, 당: ${r.sugar}g)';
      }).join('\n');

      final prompt = '''
다음은 사용자의 오늘 식사 기록입니다:

$foodList

영양 요약:
- 총 칼로리: ${summary.totalCalories}kcal (권장: ${DailyNutritionSummary.recommendedCalories}kcal)
- 총 단백질: ${summary.totalProtein.toStringAsFixed(1)}g (권장: ${DailyNutritionSummary.recommendedProtein}g)
- 총 탄수화물: ${summary.totalCarbs.toStringAsFixed(1)}g (권장: ${DailyNutritionSummary.recommendedCarbs}g)
- 총 지방: ${summary.totalFat.toStringAsFixed(1)}g (권장: ${DailyNutritionSummary.recommendedFat}g)
- 총 나트륨: ${summary.totalSodium.toStringAsFixed(1)}mg (권장: ${DailyNutritionSummary.recommendedSodium}mg)
- 총 당: ${summary.totalSugar.toStringAsFixed(1)}g (권장: ${DailyNutritionSummary.recommendedSugar}g)

위 정보를 바탕으로 다음을 제공해주세요:
1. 영양 섭취 분석 (권장량 대비 초과/부족한 영양소)
2. 구체적이고 실천 가능한 식단 개선 제안 (예: "점심과 저녁에 나트륨을 줄여보세요", "자기 전에 물을 많이 마셔보세요")
3. 긍정적인 측면이 있다면 칭찬도 포함

답변은 친근하고 격려하는 톤으로, 3-5개의 짧은 문장으로 작성해주세요.
''';

      final response = await http.post(
        Uri.parse(apiUrl),
        headers: {
          'Content-Type': 'application/json',
          'x-api-key': apiKey,
          'anthropic-version': '2023-06-01',
        },
        body: jsonEncode({
          'model': 'claude-3-5-sonnet-20241022',
          'max_tokens': 500,
          'messages': [
            {
              'role': 'user',
              'content': prompt,
            }
          ],
        }),
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(utf8.decode(response.bodyBytes));
        final content = data['content'][0]['text'] as String;
        return content;
      } else {
        print('API Error: ${response.statusCode} - ${response.body}');
        return _generateRuleBasedFeedback(records, summary);
      }
    } catch (e) {
      print('Error generating feedback: $e');
      return _generateRuleBasedFeedback(records, summary);
    }
  }

  // API 실패 시 규칙 기반 피드백 생성
  static String _generateRuleBasedFeedback(
    List<FoodRecord> records,
    DailyNutritionSummary summary,
  ) {
    final feedbackList = <String>[];

    // 칼로리 체크
    if (summary.isCaloriesExceeded()) {
      final excess = summary.totalCalories - DailyNutritionSummary.recommendedCalories;
      feedbackList.add('⚠️ 오늘 일일 권장 칼로리를 ${excess}kcal 초과하셨어요.');
    } else if (summary.totalCalories < DailyNutritionSummary.recommendedCalories * 0.7) {
      feedbackList.add('💡 오늘 칼로리 섭취가 부족해요. 영양가 있는 간식을 추가해보세요.');
    } else {
      feedbackList.add('✅ 칼로리 섭취가 적절해요!');
    }

    // 나트륨 체크
    if (summary.isSodiumExceeded()) {
      final excess = (summary.totalSodium - DailyNutritionSummary.recommendedSodium).toInt();
      feedbackList.add('🧂 나트륨을 ${excess}mg 초과하셨어요. 점심과 저녁에 짠 음식을 줄여보는 건 어떨까요?');
      feedbackList.add('💧 자기 전에 물을 충분히 마셔주세요. 나트륨 배출에 도움이 됩니다.');
    }

    // 당 체크
    if (summary.isSugarExceeded()) {
      final excess = (summary.totalSugar - DailyNutritionSummary.recommendedSugar).toInt();
      feedbackList.add('🍬 당 섭취가 ${excess}g 초과되었어요. 단 음료나 디저트를 줄여보세요.');
    }

    // 단백질 체크
    if (summary.totalProtein < DailyNutritionSummary.recommendedProtein * 0.7) {
      feedbackList.add('🥩 단백질 섭취가 부족해요. 살코기, 생선, 계란, 콩류를 추가해보세요.');
    } else if (summary.totalProtein >= DailyNutritionSummary.recommendedProtein) {
      feedbackList.add('💪 단백질 섭취가 충분해요!');
    }

    // 지방 체크
    if (summary.isFatExceeded()) {
      feedbackList.add('🍟 지방 섭취가 많아요. 튀김이나 기름진 음식을 줄이고 찜이나 구이로 바꿔보세요.');
    }

    // 식사 패턴 분석
    final mealTypes = records.map((r) => r.mealType).toSet();
    if (!mealTypes.contains('breakfast')) {
      feedbackList.add('🌅 아침 식사를 거르셨네요. 간단하게라도 아침을 먹으면 대사가 활발해져요.');
    }

    if (feedbackList.isEmpty) {
      feedbackList.add('👍 오늘 식단이 균형잡혀 있어요! 이대로 유지하시면 좋겠어요.');
    }

    return feedbackList.join('\n\n');
  }

  // 특정 영양소에 대한 간단한 팁
  static String getTipForNutrient(String nutrient) {
    final tips = {
      'sodium': '나트륨이 높을 때:\n- 가공식품 줄이기\n- 국물 적게 먹기\n- 채소와 과일 섭취로 칼륨 보충\n- 물 많이 마시기',
      'sugar': '당 섭취가 높을 때:\n- 단 음료 대신 물이나 차\n- 과일도 적당량만\n- 식사 후 바로 양치\n- 천천히 먹기',
      'calories': '칼로리가 높을 때:\n- 간식 줄이기\n- 식사량 조절\n- 채소 비중 늘리기\n- 가벼운 운동 추가',
      'protein': '단백질이 부족할 때:\n- 살코기, 생선 추가\n- 계란 요리\n- 두부, 콩류 활용\n- 저지방 유제품',
    };
    return tips[nutrient] ?? '균형잡힌 식단을 유지하세요!';
  }
}
