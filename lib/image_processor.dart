import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:image/image.dart' as img;
import 'dart:math' as math;

/// 이미지 처리 관련 유틸리티 클래스
class ImageProcessor {
  /// 파일에서 이미지를 로드
  static Future<img.Image?> loadImage(String path) async {
    try {
      final bytes = await File(path).readAsBytes();
      return img.decodeImage(bytes);
    } catch (e) {
      debugPrint('이미지 로드 실패: $e');
      return null;
    }
  }

  /// 이미지를 파일로 저장
  static Future<bool> saveImage(img.Image image, String path) async {
    try {
      final bytes = img.encodeJpg(image, quality: 95);
      await File(path).writeAsBytes(bytes);
      return true;
    } catch (e) {
      debugPrint('이미지 저장 실패: $e');
      return false;
    }
  }

  /// 밝기 조정 (-100 ~ 100, 0이 원본)
  static img.Image adjustBrightness(img.Image original, double percentage) {
    final image = img.Image.from(original);
    final adjustment = (percentage / 100 * 255).round();

    for (int y = 0; y < image.height; y++) {
      for (int x = 0; x < image.width; x++) {
        final pixel = image.getPixel(x, y);
        final r = _clamp(pixel.r.toInt() + adjustment);
        final g = _clamp(pixel.g.toInt() + adjustment);
        final b = _clamp(pixel.b.toInt() + adjustment);

        image.setPixelRgba(x, y, r, g, b, pixel.a.toInt());
      }
    }
    return image;
  }

  /// 대비 조정 (0.5 ~ 2.0, 1.0이 원본)
  static img.Image adjustContrast(img.Image original, double factor) {
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

  /// 채도 조정 (0.0 ~ 2.0, 1.0이 원본)
  static img.Image adjustSaturation(img.Image original, double factor) {
    final image = img.Image.from(original);

    for (int y = 0; y < image.height; y++) {
      for (int x = 0; x < image.width; x++) {
        final pixel = image.getPixel(x, y);

        // RGB를 HSL로 변환
        final hsl = _rgbToHsl(
          pixel.r.toInt(),
          pixel.g.toInt(),
          pixel.b.toInt()
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

  /// 복합 색감 보정 (밝기, 대비, 채도를 한번에 적용)
  static img.Image adjustColors(
    img.Image original, {
    double brightness = 0.0,
    double contrast = 1.0,
    double saturation = 1.0,
  }) {
    img.Image result = img.Image.from(original);

    if (brightness != 0.0) {
      result = adjustBrightness(result, brightness);
    }
    if (contrast != 1.0) {
      result = adjustContrast(result, contrast);
    }
    if (saturation != 1.0) {
      result = adjustSaturation(result, saturation);
    }

    return result;
  }

  /// 여러 이미지 변형 생성
  static Future<List<img.Image>> generateAugmentedImages(
    img.Image original,
    Function(double)? onProgress,
  ) async {
    final List<img.Image> augmented = [];

    // 밝기 조정: -20%, 0%, +20%
    final brightnessLevels = [-20.0, 0.0, 20.0];
    // 대비 조정: 0.8, 1.0, 1.2
    final contrastLevels = [0.8, 1.0, 1.2];
    // 채도 조정: 0.8, 1.0, 1.2
    final saturationLevels = [0.8, 1.0, 1.2];

    int totalCombinations = brightnessLevels.length *
                           contrastLevels.length *
                           saturationLevels.length;
    int current = 0;

    for (var brightness in brightnessLevels) {
      for (var contrast in contrastLevels) {
        for (var saturation in saturationLevels) {
          // 색감 보정 적용
          final augmentedImage = adjustColors(
            original,
            brightness: brightness,
            contrast: contrast,
            saturation: saturation,
          );

          augmented.add(augmentedImage);

          current++;
          if (onProgress != null) {
            onProgress(current / totalCombinations);
          }
        }
      }
    }

    debugPrint('생성된 증강 이미지 수: ${augmented.length}');
    return augmented;
  }

  /// 최빈값 기반 이미지 합성
  static Future<img.Image?> synthesizeByMode(
    List<img.Image> images,
    Function(double)? onProgress,
  ) async {
    if (images.isEmpty) return null;

    final width = images[0].width;
    final height = images[0].height;
    final result = img.Image(width: width, height: height);

    int totalPixels = width * height;
    int processedPixels = 0;

    // 각 픽셀 위치별로 최빈값 계산
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        // 해당 위치의 모든 이미지에서 RGB 값 수집
        List<int> rValues = [];
        List<int> gValues = [];
        List<int> bValues = [];

        for (var image in images) {
          final pixel = image.getPixel(x, y);
          rValues.add(pixel.r.toInt());
          gValues.add(pixel.g.toInt());
          bValues.add(pixel.b.toInt());
        }

        // 각 채널별 최빈값 계산
        final rMode = _calculateMode(rValues);
        final gMode = _calculateMode(gValues);
        final bMode = _calculateMode(bValues);

        // 결과 이미지에 설정
        result.setPixelRgba(x, y, rMode, gMode, bMode, 255);

        processedPixels++;
        if (onProgress != null && processedPixels % 1000 == 0) {
          onProgress(processedPixels / totalPixels);
        }
      }
    }

    if (onProgress != null) {
      onProgress(1.0);
    }

    return result;
  }

  /// 최빈값 계산
  static int _calculateMode(List<int> values) {
    if (values.isEmpty) return 0;

    // 빈도수 계산
    Map<int, int> frequency = {};
    for (var value in values) {
      frequency[value] = (frequency[value] ?? 0) + 1;
    }

    // 최대 빈도수 찾기
    int maxFrequency = 0;
    int mode = values[0];

    frequency.forEach((value, count) {
      if (count > maxFrequency) {
        maxFrequency = count;
        mode = value;
      } else if (count == maxFrequency) {
        // 동일한 빈도수인 경우 평균값 사용
        mode = ((mode + value) / 2).round();
      }
    });

    return mode;
  }

  /// 값을 0-255 범위로 제한
  static int _clamp(int value) {
    return value.clamp(0, 255);
  }

  /// RGB를 HSL로 변환
  static List<double> _rgbToHsl(int r, int g, int b) {
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
  static List<int> _hslToRgb(double h, double s, double l) {
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

  /// Hue를 RGB 값으로 변환 (HSL 변환용 헬퍼)
  static double _hueToRgb(double p, double q, double t) {
    double td = t;
    if (td < 0) td += 1;
    if (td > 1) td -= 1;
    if (td < 1 / 6) return p + (q - p) * 6 * td;
    if (td < 1 / 2) return q;
    if (td < 2 / 3) return p + (q - p) * (2 / 3 - td) * 6;
    return p;
  }
}
