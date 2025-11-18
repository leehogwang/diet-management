import '../database/database_helper.dart';
import '../services/food_nutrition_api.dart';
import '../models/meal_nutrition.dart';

/// 데이터베이스의 모든 식단 데이터를 자동으로 후보정하는 유틸리티
class BatchCorrection {
  /// 모든 식단 데이터를 API로 검색하여 영양 정보 자동 보정
  ///
  /// [dryRun]: true로 설정하면 실제 업데이트 없이 시뮬레이션만 수행
  /// [delayMs]: 각 API 호출 사이의 지연 시간 (밀리초, 기본 500ms)
  static Future<CorrectionResult> correctAllMeals({
    bool dryRun = false,
    int delayMs = 500,
  }) async {
    final result = CorrectionResult();

    try {
      // 모든 식단 데이터 조회
      final meals = await DatabaseHelper.instance.readAll();
      result.totalCount = meals.length;

      print('총 ${meals.length}개의 식단 데이터를 처리합니다.');
      print('DryRun 모드: ${dryRun ? "예 (실제 업데이트 없음)" : "아니오"}');
      print('-------------------------------------------');

      for (var i = 0; i < meals.length; i++) {
        final meal = meals[i];
        print('\n[${i + 1}/${meals.length}] ${meal.foodName} 처리 중...');

        try {
          // API로 식품 검색
          final searchResults = await FoodNutritionAPI.searchFood(meal.foodName);

          if (searchResults.isEmpty) {
            print('  ❌ 검색 결과 없음');
            result.notFoundList.add(meal.foodName);
            continue;
          }

          // 첫 번째 검색 결과 사용
          final apiData = searchResults.first;

          print('  ✓ 검색 성공: ${apiData.foodName}');
          print('    기존: ${meal.calories}kcal, 탄${meal.carbohydrates}g, 단${meal.protein}g, 지${meal.fat}g');
          print('    변경: ${apiData.calories}kcal, 탄${apiData.carbohydrates}g, 단${apiData.protein}g, 지${apiData.fat}g');

          if (!dryRun) {
            // 실제 업데이트 수행
            final updatedMeal = meal.copyWith(
              calories: apiData.calories,
              carbohydrates: apiData.carbohydrates,
              protein: apiData.protein,
              fat: apiData.fat,
            );
            await DatabaseHelper.instance.update(updatedMeal);
            print('  💾 데이터베이스 업데이트 완료');
          }

          result.successCount++;
          result.successList.add(meal.foodName);

          // API 호출 제한 방지를 위한 지연
          if (i < meals.length - 1) {
            await Future.delayed(Duration(milliseconds: delayMs));
          }
        } catch (e) {
          print('  ⚠️ 오류 발생: $e');
          result.errorList.add('${meal.foodName}: $e');
        }
      }

      print('\n===========================================');
      print('처리 완료!');
      print('총 처리: ${result.totalCount}개');
      print('성공: ${result.successCount}개');
      print('검색 결과 없음: ${result.notFoundList.length}개');
      print('오류: ${result.errorList.length}개');
      print('===========================================\n');

      if (result.notFoundList.isNotEmpty) {
        print('\n검색 결과가 없는 항목:');
        for (var name in result.notFoundList) {
          print('  - $name');
        }
      }

      if (result.errorList.isNotEmpty) {
        print('\n오류가 발생한 항목:');
        for (var error in result.errorList) {
          print('  - $error');
        }
      }
    } catch (e) {
      print('배치 작업 중 치명적 오류 발생: $e');
      result.fatalError = e.toString();
    }

    return result;
  }

  /// 특정 음식만 선택적으로 후보정
  static Future<bool> correctSingleMeal(int mealId) async {
    try {
      final meal = await DatabaseHelper.instance.read(mealId);
      if (meal == null) {
        print('ID $mealId 식단 데이터를 찾을 수 없습니다.');
        return false;
      }

      print('${meal.foodName} 후보정 시작...');

      final searchResults = await FoodNutritionAPI.searchFood(meal.foodName);
      if (searchResults.isEmpty) {
        print('검색 결과가 없습니다.');
        return false;
      }

      final apiData = searchResults.first;
      final updatedMeal = meal.copyWith(
        calories: apiData.calories,
        carbohydrates: apiData.carbohydrates,
        protein: apiData.protein,
        fat: apiData.fat,
      );

      await DatabaseHelper.instance.update(updatedMeal);
      print('✓ 후보정 완료!');
      return true;
    } catch (e) {
      print('오류 발생: $e');
      return false;
    }
  }

  /// 검색 결과가 없는 항목들의 목록 확인
  static Future<List<String>> findMealsWithoutApiMatch() async {
    final notFoundList = <String>[];
    final meals = await DatabaseHelper.instance.readAll();

    print('검색 결과 확인 중...');
    for (var i = 0; i < meals.length; i++) {
      final meal = meals[i];
      print('[${i + 1}/${meals.length}] ${meal.foodName} 확인 중...');

      try {
        final results = await FoodNutritionAPI.searchFood(meal.foodName);
        if (results.isEmpty) {
          notFoundList.add(meal.foodName);
          print('  ❌ 검색 결과 없음');
        } else {
          print('  ✓ 검색 결과 있음');
        }
        await Future.delayed(const Duration(milliseconds: 500));
      } catch (e) {
        print('  ⚠️ 오류: $e');
      }
    }

    print('\n검색 결과가 없는 항목 (${notFoundList.length}개):');
    for (var name in notFoundList) {
      print('  - $name');
    }

    return notFoundList;
  }
}

/// 후보정 결과 정보
class CorrectionResult {
  int totalCount = 0;
  int successCount = 0;
  List<String> successList = [];
  List<String> notFoundList = [];
  List<String> errorList = [];
  String? fatalError;

  bool get hasErrors => errorList.isNotEmpty || fatalError != null;
  bool get isSuccess => successCount == totalCount && !hasErrors;
}
