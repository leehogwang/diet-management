import 'package:flutter/material.dart';
import 'dart:io';
import 'dart:convert';
import 'package:path_provider/path_provider.dart';

class NutritionData {
  final DateTime date;
  final double calories;
  final double sodium;
  final double sugar;
  final double carbohydrates;
  final String? imagePath;

  NutritionData({
    required this.date,
    required this.calories,
    required this.sodium,
    required this.sugar,
    required this.carbohydrates,
    this.imagePath,
  });

  // JSON에서 NutritionData로 변환
  factory NutritionData.fromJson(Map<String, dynamic> json) {
    return NutritionData(
      date: DateTime.parse(json['date']),
      calories: json['calories']?.toDouble() ?? 0.0,
      sodium: json['sodium']?.toDouble() ?? 0.0,
      sugar: json['sugar']?.toDouble() ?? 0.0,
      carbohydrates: json['carbohydrates']?.toDouble() ?? 0.0,
      imagePath: json['imagePath'],
    );
  }

  // NutritionData를 JSON으로 변환
  Map<String, dynamic> toJson() {
    return {
      'date': date.toIso8601String(),
      'calories': calories,
      'sodium': sodium,
      'sugar': sugar,
      'carbohydrates': carbohydrates,
      'imagePath': imagePath,
    };
  }

  // 실제 사진 분석 데이터 로드
  static Future<List<NutritionData>> loadRealData() async {
    try {
      final directory = await getApplicationDocumentsDirectory();
      final nutritionFile = File('${directory.path}/nutrition_data.json');

      if (!await nutritionFile.exists()) {
        return [];
      }

      final jsonString = await nutritionFile.readAsString();
      final List<dynamic> jsonData = jsonDecode(jsonString);

      return jsonData
          .map((item) => NutritionData.fromJson(item))
          .toList()
          ..sort((a, b) => b.date.compareTo(a.date)); // 최신순 정렬
    } catch (e) {
      debugPrint('Error loading nutrition data: $e');
      return [];
    }
  }

  // 영양 데이터 저장
  static Future<void> saveNutritionData({
    required DateTime date,
    required double calories,
    required double sodium,
    required double sugar,
    required double carbohydrates,
    String? imagePath,
  }) async {
    try {
      final directory = await getApplicationDocumentsDirectory();
      final nutritionFile = File('${directory.path}/nutrition_data.json');

      List<NutritionData> existingData = [];

      // 기존 데이터 로드
      if (await nutritionFile.exists()) {
        final jsonString = await nutritionFile.readAsString();
        final List<dynamic> jsonData = jsonDecode(jsonString);
        existingData = jsonData.map((item) => NutritionData.fromJson(item)).toList();
      }

      // 새 데이터 추가
      final newData = NutritionData(
        date: date,
        calories: calories,
        sodium: sodium,
        sugar: sugar,
        carbohydrates: carbohydrates,
        imagePath: imagePath,
      );

      existingData.add(newData);

      // JSON으로 저장
      final jsonString = jsonEncode(existingData.map((data) => data.toJson()).toList());
      await nutritionFile.writeAsString(jsonString);

      debugPrint('✅ Nutrition data saved successfully');
    } catch (e) {
      debugPrint('❌ Error saving nutrition data: $e');
    }
  }

  // 샘플 데이터 생성 (테스트용 - 이제는 사용하지 않음)
  static List<NutritionData> generateSampleData() {
    final now = DateTime.now();
    final List<NutritionData> data = [];

    for (int i = 0; i < 30; i++) {
      final date = now.subtract(Duration(days: i));
      data.add(NutritionData(
        date: date,
        calories: 1800 + (i % 5) * 100,
        sodium: 1500 + (i % 8) * 50,
        sugar: 30 + (i % 4) * 5,
        carbohydrates: 250 + (i % 6) * 10,
      ));
    }

    return data;
  }
}