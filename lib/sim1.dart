import 'dart:io';
import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:image/image.dart' as img;
import 'package:http/http.dart' as http;
import 'dart:math' as math;

/// 간단한 이미지 증강 및 LLM 평가 시스템
///
/// 사용법:
/// ```dart
/// final evaluator = SimpleImageEvaluator(apiKey: 'your-openai-api-key');
/// await evaluator.evaluateImage('path/to/food.jpg', '비빔밥');
/// ```

class SimpleImageEvaluator {
  final String apiKey;
  final String model;
  static const String baseUrl = 'https://api.openai.com/v1/chat/completions';

  SimpleImageEvaluator({
    required this.apiKey,
    this.model = 'gpt-4o',
  });

  /// 단일 이미지 평가 메인 함수
  Future<void> evaluateImage(String imagePath, String groundTruth) async {
    print('\n════════════════════════════════════════════════════════');
    print('🍴 음식 이미지 증강 평가 시스템');
    print('════════════════════════════════════════════════════════\n');

    print('📁 이미지 로딩: $imagePath');
    print('🎯 정답 음식명: $groundTruth\n');

    try {
      // [0%] 이미지 로드
      print('[0%] 이미지 로드 중...');
      final originalImage = await _loadImage(imagePath);
      if (originalImage == null) {
        print('❌ 오류: 이미지를 로드할 수 없습니다.');
        return;
      }

      // 결과 저장용
      final Map<String, dynamic> results = {
        'ground_truth': groundTruth,
        'evaluations': {},
      };

      // [10%] 원본 이미지 예측
      print('[10%] 원본 이미지 분석 중...');
      final originalPrediction = await _predictFood(originalImage, '원본');
      results['evaluations']['original'] = {
        'predicted': originalPrediction,
      };
      print('✅ 원본 예측: $originalPrediction\n');

      // [20%] Contrast 1.4 적용
      print('[20%] Contrast 1.4 적용 중...');
      final contrastImage = _adjustContrast(originalImage, 1.4);

      // [40%] Contrast 이미지 예측
      print('[40%] Contrast 1.4 이미지 분석 중...');
      final contrastPrediction = await _predictFood(contrastImage, 'Contrast 1.4');
      results['evaluations']['contrast_1.4'] = {
        'predicted': contrastPrediction,
      };
      print('✅ Contrast 1.4 예측: $contrastPrediction\n');

      // [50%] Saturation 1.4 적용
      print('[50%] Saturation 1.4 적용 중...');
      final saturationImage = _adjustSaturation(originalImage, 1.4);

      // [60%] Saturation 이미지 예측
      print('[60%] Saturation 1.4 이미지 분석 중...');
      final saturationPrediction = await _predictFood(saturationImage, 'Saturation 1.4');
      results['evaluations']['saturation_1.4'] = {
        'predicted': saturationPrediction,
      };
      print('✅ Saturation 1.4 예측: $saturationPrediction\n');

      // [70%] LLM에게 원본 vs Contrast 비교 요청
      print('[70%] LLM이 원본 vs Contrast 차이 분석 중...');
      final contrastComparison = await _compareImages(
        originalImage,
        contrastImage,
        groundTruth,
        originalPrediction,
        contrastPrediction,
        'Contrast 1.4',
      );
      results['evaluations']['contrast_1.4']['comparison'] = contrastComparison;
      print('✅ Contrast 비교 완료\n');

      // [85%] LLM에게 원본 vs Saturation 비교 요청
      print('[85%] LLM이 원본 vs Saturation 차이 분석 중...');
      final saturationComparison = await _compareImages(
        originalImage,
        saturationImage,
        groundTruth,
        originalPrediction,
        saturationPrediction,
        'Saturation 1.4',
      );
      results['evaluations']['saturation_1.4']['comparison'] = saturationComparison;
      print('✅ Saturation 비교 완료\n');

      // [100%] 결과 출력
      print('[100%] 평가 완료!\n');
      _printDetailedResults(results);

    } catch (e, stackTrace) {
      print('❌ 오류 발생: $e');
      print('스택 트레이스: $stackTrace');
    }
  }

  /// 이미지 파일 로드
  Future<img.Image?> _loadImage(String path) async {
    try {
      final bytes = await File(path).readAsBytes();
      return img.decodeImage(bytes);
    } catch (e) {
      debugPrint('이미지 로드 실패: $e');
      return null;
    }
  }

  /// 대비 조정 (메모리 기반)
  img.Image _adjustContrast(img.Image original, double factor) {
    final image = img.Image.from(original);
    final contrast = (factor - 1.0) * 128;

    for (int y = 0; y < image.height; y++) {
      for (int x = 0; x < image.width; x++) {
        final pixel = image.getPixel(x, y);
        final r = _clamp(((pixel.r - 128) * factor + 128 + contrast).round());
        final g = _clamp(((pixel.g - 128) * factor + 128 + contrast).round());
        final b = _clamp(((pixel.b - 128) * factor + 128 + contrast).round());

        image.setPixelRgba(x, y, r, g, b, pixel.a.toInt());
      }
    }
    return image;
  }

  /// 채도 조정 (메모리 기반)
  img.Image _adjustSaturation(img.Image original, double factor) {
    final image = img.Image.from(original);

    for (int y = 0; y < image.height; y++) {
      for (int x = 0; x < image.width; x++) {
        final pixel = image.getPixel(x, y);

        // RGB를 HSL로 변환
        final hsl = _rgbToHsl(
          pixel.r.toInt(),
          pixel.g.toInt(),
          pixel.b.toInt(),
        );

        // 채도 조정
        hsl[1] = _clamp((hsl[1] * factor * 100).round()) / 100.0;

        // HSL을 RGB로 변환
        final rgb = _hslToRgb(hsl[0], hsl[1], hsl[2]);

        image.setPixelRgba(x, y, rgb[0], rgb[1], rgb[2], pixel.a.toInt());
      }
    }
    return image;
  }

  /// OpenAI API를 통해 음식 예측
  Future<String> _predictFood(img.Image image, String label) async {
    try {
      // Base64 인코딩 (API 호출 직전에만 수행)
      final bytes = img.encodePng(image);
      final base64Image = base64Encode(bytes);

      final response = await http.post(
        Uri.parse(baseUrl),
        headers: {
          'Content-Type': 'application/json',
          'Authorization': 'Bearer $apiKey',
        },
        body: jsonEncode({
          'model': model,
          'messages': [
            {
              'role': 'user',
              'content': [
                {
                  'type': 'text',
                  'text': '이 이미지에 있는 음식의 이름을 한국어로 정확히 말해주세요. 음식 이름만 답변하고 다른 설명은 하지 마세요.',
                },
                {
                  'type': 'image_url',
                  'image_url': {
                    'url': 'data:image/png;base64,$base64Image',
                  },
                },
              ],
            },
          ],
          'max_tokens': 100,
        }),
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(utf8.decode(response.bodyBytes));
        return data['choices'][0]['message']['content'].toString().trim();
      } else {
        debugPrint('API 오류: ${response.statusCode} - ${response.body}');
        return 'API_ERROR';
      }
    } catch (e) {
      debugPrint('API 호출 실패: $e');
      return 'ERROR';
    }
  }

  /// LLM을 통해 두 이미지 비교 및 오차 분석
  Future<Map<String, dynamic>> _compareImages(
    img.Image originalImage,
    img.Image augmentedImage,
    String groundTruth,
    String originalPrediction,
    String augmentedPrediction,
    String augmentationType,
  ) async {
    try {
      // Base64 인코딩 (API 호출 직전에만)
      final originalBase64 = base64Encode(img.encodePng(originalImage));
      final augmentedBase64 = base64Encode(img.encodePng(augmentedImage));

      final prompt = '''
다음 두 음식 이미지를 비교하여 분석해주세요:

**정답 음식명**: $groundTruth
**원본 이미지 예측**: $originalPrediction
**$augmentationType 적용 이미지 예측**: $augmentedPrediction

다음 형식의 JSON으로만 답변해주세요 (다른 설명 없이):
{
  "error_score": (0~100 사이의 숫자, 0에 가까울수록 정답에 가까움),
  "visual_differences": "(색상, 밝기, 선명도 등의 시각적 차이점 설명)",
  "semantic_difference": "(두 예측 결과의 의미적 차이 설명)",
  "recommendation": "(어느 쪽이 더 정답에 가까운지, 왜 그런지 설명)"
}
''';

      final response = await http.post(
        Uri.parse(baseUrl),
        headers: {
          'Content-Type': 'application/json',
          'Authorization': 'Bearer $apiKey',
        },
        body: jsonEncode({
          'model': model,
          'messages': [
            {
              'role': 'user',
              'content': [
                {
                  'type': 'text',
                  'text': '원본 이미지:',
                },
                {
                  'type': 'image_url',
                  'image_url': {
                    'url': 'data:image/png;base64,$originalBase64',
                  },
                },
                {
                  'type': 'text',
                  'text': '$augmentationType 적용 이미지:',
                },
                {
                  'type': 'image_url',
                  'image_url': {
                    'url': 'data:image/png;base64,$augmentedBase64',
                  },
                },
                {
                  'type': 'text',
                  'text': prompt,
                },
              ],
            },
          ],
          'max_tokens': 1000,
        }),
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(utf8.decode(response.bodyBytes));
        final content = data['choices'][0]['message']['content'].toString().trim();

        // JSON 추출 (마크다운 코드 블록 제거)
        String jsonStr = content;
        if (content.contains('```json')) {
          jsonStr = content.split('```json')[1].split('```')[0].trim();
        } else if (content.contains('```')) {
          jsonStr = content.split('```')[1].split('```')[0].trim();
        }

        try {
          return jsonDecode(jsonStr);
        } catch (e) {
          // JSON 파싱 실패 시 기본 응답
          return {
            'error_score': 50,
            'visual_differences': content,
            'semantic_difference': '파싱 실패',
            'recommendation': '수동 확인 필요',
          };
        }
      } else {
        debugPrint('비교 API 오류: ${response.statusCode} - ${response.body}');
        return {
          'error_score': 999,
          'visual_differences': 'API 오류',
          'semantic_difference': 'API 호출 실패',
          'recommendation': 'API 상태 확인 필요',
        };
      }
    } catch (e) {
      debugPrint('비교 분석 실패: $e');
      return {
        'error_score': 999,
        'visual_differences': '분석 실패',
        'semantic_difference': e.toString(),
        'recommendation': '재시도 필요',
      };
    }
  }

  /// 상세 결과 출력
  void _printDetailedResults(Map<String, dynamic> results) {
    print('════════════════════════════════════════════════════════');
    print('📊 평가 결과 상세');
    print('════════════════════════════════════════════════════════\n');

    print('🎯 정답: ${results['ground_truth']}\n');

    // 원본 결과
    print('─────────────────────────────────────');
    print('📷 원본 이미지');
    print('─────────────────────────────────────');
    print('예측: ${results['evaluations']['original']['predicted']}\n');

    // Contrast 결과
    print('─────────────────────────────────────');
    print('🔆 Contrast 1.4 적용');
    print('─────────────────────────────────────');
    final contrastEval = results['evaluations']['contrast_1.4'];
    print('예측: ${contrastEval['predicted']}');
    if (contrastEval['comparison'] != null) {
      final comp = contrastEval['comparison'];
      print('오차 점수: ${comp['error_score']}/100');
      print('시각적 차이: ${comp['visual_differences']}');
      print('의미적 차이: ${comp['semantic_difference']}');
      print('추천: ${comp['recommendation']}');
    }
    print('');

    // Saturation 결과
    print('─────────────────────────────────────');
    print('🎨 Saturation 1.4 적용');
    print('─────────────────────────────────────');
    final saturationEval = results['evaluations']['saturation_1.4'];
    print('예측: ${saturationEval['predicted']}');
    if (saturationEval['comparison'] != null) {
      final comp = saturationEval['comparison'];
      print('오차 점수: ${comp['error_score']}/100');
      print('시각적 차이: ${comp['visual_differences']}');
      print('의미적 차이: ${comp['semantic_difference']}');
      print('추천: ${comp['recommendation']}');
    }
    print('');

    // 최종 추천
    print('─────────────────────────────────────');
    print('✨ 최종 분석');
    print('─────────────────────────────────────');

    final contrastScore = contrastEval['comparison']?['error_score'] ?? 999;
    final saturationScore = saturationEval['comparison']?['error_score'] ?? 999;

    if (contrastScore < saturationScore) {
      print('🏆 Contrast 1.4가 더 정확합니다 (오차: $contrastScore)');
    } else if (saturationScore < contrastScore) {
      print('🏆 Saturation 1.4가 더 정확합니다 (오차: $saturationScore)');
    } else {
      print('🤝 두 조정 모두 비슷한 정확도를 보입니다');
    }
    print('');

    // 전체 JSON 출력
    print('════════════════════════════════════════════════════════');
    print('📄 전체 결과 JSON');
    print('════════════════════════════════════════════════════════');
    const encoder = JsonEncoder.withIndent('  ');
    print(encoder.convert(results));
    print('════════════════════════════════════════════════════════\n');
  }

  // ======================================================================
  // 유틸리티 함수들
  // ======================================================================

  /// 값을 0-255 범위로 제한
  int _clamp(int value) {
    return value.clamp(0, 255);
  }

  /// RGB를 HSL로 변환
  List<double> _rgbToHsl(int r, int g, int b) {
    final rd = r / 255.0;
    final gd = g / 255.0;
    final bd = b / 255.0;

    final max = math.max(rd, math.max(gd, bd));
    final min = math.min(rd, math.min(gd, bd));
    final delta = max - min;

    double h = 0.0;
    double s = 0.0;
    double l = (max + min) / 2.0;

    if (delta != 0) {
      s = l < 0.5 ? delta / (max + min) : delta / (2.0 - max - min);

      if (max == rd) {
        h = ((gd - bd) / delta + (gd < bd ? 6 : 0)) / 6.0;
      } else if (max == gd) {
        h = ((bd - rd) / delta + 2) / 6.0;
      } else {
        h = ((rd - gd) / delta + 4) / 6.0;
      }
    }

    return [h, s, l];
  }

  /// HSL을 RGB로 변환
  List<int> _hslToRgb(double h, double s, double l) {
    double r, g, b;

    if (s == 0) {
      r = g = b = l;
    } else {
      final q = l < 0.5 ? l * (1 + s) : l + s - l * s;
      final p = 2 * l - q;
      r = _hueToRgb(p, q, h + 1 / 3);
      g = _hueToRgb(p, q, h);
      b = _hueToRgb(p, q, h - 1 / 3);
    }

    return [
      (r * 255).round().clamp(0, 255),
      (g * 255).round().clamp(0, 255),
      (b * 255).round().clamp(0, 255),
    ];
  }

  /// Hue를 RGB 값으로 변환
  double _hueToRgb(double p, double q, double t) {
    double td = t;
    if (td < 0) td += 1;
    if (td > 1) td -= 1;
    if (td < 1 / 6) return p + (q - p) * 6 * td;
    if (td < 1 / 2) return q;
    if (td < 2 / 3) return p + (q - p) * (2 / 3 - td) * 6;
    return p;
  }
}

// ======================================================================
// 사용 예제 (main 함수)
// ======================================================================

/// 사용 예제
void main() async {
  // OpenAI API 키 설정 (환경 변수에서 읽기)
  const apiKey = String.fromEnvironment('OPENAI_API_KEY', defaultValue: '');

  // 평가자 인스턴스 생성
  final evaluator = SimpleImageEvaluator(
    apiKey: apiKey,
    model: 'gpt-4o', // 또는 'gpt-4o-mini'
  );

  // 단일 이미지 평가
  // 사용자는 아래 경로를 실제 음식 이미지 경로로 변경
  await evaluator.evaluateImage(
    'path/to/your/food_image.jpg',  // ← 여기에 실제 이미지 경로 입력
    '비빔밥',                         // ← 여기에 정답 음식명 입력
  );

  print('\n✅ 평가 완료! 위 결과를 확인하세요.\n');

  // 추가 이미지를 평가하려면:
  // await evaluator.evaluateImage('path/to/another_image.jpg', '김치찌개');
  // await evaluator.evaluateImage('path/to/third_image.jpg', '불고기');
}
