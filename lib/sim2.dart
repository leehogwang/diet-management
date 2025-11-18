import 'package:flutter/foundation.dart';
import 'image_processor.dart';

/// image_processor.dart 사용 예제
///
/// 이 파일은 image_processor의 주요 기능을 사용하는 방법을 보여줍니다.
///
/// 사용 전 준비사항:
/// 1. pubspec.yaml에 다음 의존성 추가:
///    - image: ^4.0.0
///    - http: ^1.0.0
///    - path: ^1.8.0
///
/// 2. .env 파일이나 설정에 OpenAI API 키 추가

class ImageProcessorExamples {

  /// 예제 1: 단일 이미지 증강 및 LLM 평가
  static Future<void> example1SingleImageEvaluation() async {
    // OpenAI API 키 설정 (실제 키로 교체 필요)
    const apiKey = '';

    // OpenAI API 인스턴스 생성
    final openaiAPI = OpenAIVisionAPI(apiKey: apiKey);

    // 평가자 인스턴스 생성
    final evaluator = ImageAugmentationEvaluator(openaiAPI: openaiAPI);

    // 단일 이미지 평가
    try {
      final report = await evaluator.evaluateSingleImage(
        'path/to/your/food_image.jpg',  // 이미지 경로
        '비빔밥',                         // 정답 음식명
        'output/augmentation_results',   // 출력 디렉토리
        (message, progress) {
          debugPrint('진행 상황: $message (${(progress * 100).toStringAsFixed(1)}%)');
        },
      );

      // 결과 출력
      ImageAugmentationEvaluator.printResults([report]);

      // JSON으로 저장
      await ImageAugmentationEvaluator.saveResultsAsJson(
        [report],
        'output/results.json',
      );

      // CSV로 저장
      await ImageAugmentationEvaluator.saveResultsAsCsv(
        [report],
        'output/results.csv',
      );

      debugPrint('최적 조합: ${report.bestCombination}');
      debugPrint('최소 오차: ${report.bestErrorScore}');

    } catch (e) {
      debugPrint('오류 발생: $e');
    }
  }

  /// 예제 2: 여러 이미지 배치 평가
  static Future<void> example2BatchImageEvaluation() async {
    // OpenAI API 키 설정
    const apiKey = 'your-openai-api-key-here';

    final openaiAPI = OpenAIVisionAPI(apiKey: apiKey);
    final evaluator = ImageAugmentationEvaluator(openaiAPI: openaiAPI);

    // 평가할 이미지 목록
    final imagesToEvaluate = [
      {
        'path': 'images/bibimbap.jpg',
        'groundTruth': '비빔밥',
      },
      {
        'path': 'images/kimchi_jjigae.jpg',
        'groundTruth': '김치찌개',
      },
      {
        'path': 'images/bulgogi.jpg',
        'groundTruth': '불고기',
      },
    ];

    try {
      // 배치 평가 실행
      final reports = await evaluator.evaluateBatchImages(
        imagesToEvaluate,
        'output/batch_results',
        (message, progress) {
          debugPrint('$message (${(progress * 100).toStringAsFixed(1)}%)');
        },
      );

      // 모든 결과 출력
      ImageAugmentationEvaluator.printResults(reports);

      // JSON으로 저장
      await ImageAugmentationEvaluator.saveResultsAsJson(
        reports,
        'output/batch_results.json',
      );

      // CSV로 저장
      await ImageAugmentationEvaluator.saveResultsAsCsv(
        reports,
        'output/batch_results.csv',
      );

      // 전체 최적 조합 분석
      final bestCombinations = reports.map((r) => r.bestCombination).toList();
      debugPrint('\n전체 이미지의 최적 조합:');
      for (var i = 0; i < reports.length; i++) {
        debugPrint('  ${reports[i].originalFileName}: ${reports[i].bestCombination}');
      }

    } catch (e) {
      debugPrint('오류 발생: $e');
    }
  }

  /// 예제 3: 증강 이미지만 생성 (LLM 평가 없이)
  static Future<void> example3GenerateAugmentedImagesOnly() async {
    try {
      // 이미지 로드
      final originalImage = await ImageProcessor.loadImage('path/to/image.jpg');
      if (originalImage == null) {
        debugPrint('이미지 로드 실패');
        return;
      }

      // 증강 이미지 생성 및 저장
      final augmentedImages = await ImageProcessor.generateAndSaveAugmentedImages(
        originalImage,
        'my_food_image.jpg',  // 원본 파일명
        'output/augmented',   // 출력 디렉토리
        (progress) {
          debugPrint('생성 진행: ${(progress * 100).toStringAsFixed(1)}%');
        },
      );

      debugPrint('총 ${augmentedImages.length}개의 증강 이미지 생성 완료');
      debugPrint('저장 경로: output/augmented/my_food_image/');

      // 각 증강 이미지 정보 출력
      for (var info in augmentedImages) {
        debugPrint('  ${info.label}: brightness=${info.brightness}, '
                  'contrast=${info.contrast}, saturation=${info.saturation}');
      }

    } catch (e) {
      debugPrint('오류 발생: $e');
    }
  }

  /// 예제 4: 개별 색감 조정 사용
  static Future<void> example4IndividualAdjustments() async {
    try {
      final originalImage = await ImageProcessor.loadImage('path/to/image.jpg');
      if (originalImage == null) {
        debugPrint('이미지 로드 실패');
        return;
      }

      // 밝기만 조정
      final brightened = ImageProcessor.adjustBrightness(originalImage, 20.0);
      await ImageProcessor.saveImage(brightened, 'output/brightened.png');

      // 대비만 조정
      final contrasted = ImageProcessor.adjustContrast(originalImage, 1.4);
      await ImageProcessor.saveImage(contrasted, 'output/contrasted.png');

      // 채도만 조정
      final saturated = ImageProcessor.adjustSaturation(originalImage, 1.4);
      await ImageProcessor.saveImage(saturated, 'output/saturated.png');

      // 모든 조정 동시 적용
      final enhanced = ImageProcessor.adjustColors(
        originalImage,
        brightness: 12.0,
        contrast: 1.2,
        saturation: 1.4,
      );
      await ImageProcessor.saveImage(enhanced, 'output/enhanced.png');

      debugPrint('모든 조정 이미지 저장 완료');

    } catch (e) {
      debugPrint('오류 발생: $e');
    }
  }

  /// 예제 5: 다른 LLM 모델 사용
  static Future<void> example5CustomLLMModel() async {
    // GPT-4o-mini 모델 사용 (더 빠르고 저렴)
    final openaiAPI = OpenAIVisionAPI(
      apiKey: 'your-api-key-here',
      model: 'gpt-4o-mini',
    );

    final evaluator = ImageAugmentationEvaluator(openaiAPI: openaiAPI);

    // ... 평가 로직은 동일
  }

  /// 예제 6: 커스텀 진행 상황 표시
  static Future<void> example6CustomProgressDisplay() async {
    const apiKey = 'your-openai-api-key-here';

    final openaiAPI = OpenAIVisionAPI(apiKey: apiKey);
    final evaluator = ImageAugmentationEvaluator(openaiAPI: openaiAPI);

    // 진행 상황을 UI에 표시하는 예제
    final report = await evaluator.evaluateSingleImage(
      'path/to/image.jpg',
      '김치찌개',
      'output/results',
      (message, progress) {
        // UI 업데이트 로직
        // setState(() {
        //   _progressMessage = message;
        //   _progressValue = progress;
        // });
        debugPrint('[${"=" * (progress * 20).toInt()}${" " * (20 - (progress * 20).toInt())}] '
                  '$message ${(progress * 100).toStringAsFixed(0)}%');
      },
    );

    debugPrint('평가 완료!');
  }
}

/// main 함수 예제
void main() async {
  debugPrint('=== 이미지 증강 및 LLM 평가 예제 ===\n');

  // 예제 1: 단일 이미지 평가
  // await ImageProcessorExamples.example1SingleImageEvaluation();

  // 예제 2: 배치 이미지 평가
  // await ImageProcessorExamples.example2BatchImageEvaluation();

  // 예제 3: 증강 이미지만 생성
  // await ImageProcessorExamples.example3GenerateAugmentedImagesOnly();

  // 예제 4: 개별 색감 조정
  // await ImageProcessorExamples.example4IndividualAdjustments();

  debugPrint('\n모든 예제 실행 완료!');
}
