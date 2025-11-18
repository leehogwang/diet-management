import 'package:flutter/widgets.dart';
import 'services/food_nutrition_api.dart';
import 'utils/batch_correction.dart';

/// 데이터 후보정 실행 스크립트
///
/// 사용 방법:
/// 1. 터미널에서 프로젝트 폴더로 이동
/// 2. flutter run -t lib/run_correction.dart 실행
void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  print('==================================================');
  print('식단 데이터 후보정 스크립트');
  print('==================================================\n');

  // API 키 설정 (발급받은 키로 변경하세요)
  const apiKey = '0348c25a8c52f6b3fd54b1e82ab0a22ca7134289610c305438667854c802aac5';
  FoodNutritionAPI.setApiKey(apiKey);

  print('API 키 검증 중...');
  final isValid = await FoodNutritionAPI.validateApiKey();
  if (!isValid) {
    print('❌ API 키가 유효하지 않습니다.');
    print('lib/run_correction.dart 파일에서 API 키를 확인하세요.');
    return;
  }
  print('✓ API 키 유효\n');

  print('작업 선택:');
  print('1. 전체 데이터 후보정 (시뮬레이션 - 실제 업데이트 안함)');
  print('2. 전체 데이터 후보정 (실제 업데이트)');
  print('3. 검색 결과 없는 항목만 확인');
  print('\n아래 코드를 수정하여 원하는 작업을 선택하세요.\n');

  // ========================================
  // 여기서 실행할 작업을 선택하세요
  // ========================================

  // 옵션 1: 시뮬레이션 (실제 업데이트 없음)
  await BatchCorrection.correctAllMeals(dryRun: true);

  // 옵션 2: 실제 업데이트 (주석 해제하여 사용)
  // await BatchCorrection.correctAllMeals(dryRun: false);

  // 옵션 3: 검색 결과 없는 항목만 확인 (주석 해제하여 사용)
  // await BatchCorrection.findMealsWithoutApiMatch();

  print('\n작업 완료! 프로그램을 종료합니다.');
}
