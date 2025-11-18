import 'package:flutter/services.dart';
import 'package:csv/csv.dart';

class FoodItem {
  final String code;
  final String name;
  final double calories;
  final double protein;
  final double fat;
  final double carbohydrates;
  final String category;
  final String? subcategory;

  FoodItem({
    required this.code,
    required this.name,
    required this.calories,
    required this.protein,
    required this.fat,
    required this.carbohydrates,
    required this.category,
    this.subcategory,
  });

  factory FoodItem.fromCsvRow(List<dynamic> row) {
    return FoodItem(
      code: row[0]?.toString() ?? '',
      name: row[1]?.toString() ?? '',
      calories: _parseDouble(row[17]),
      protein: _parseDouble(row[19]),
      fat: _parseDouble(row[20]),
      carbohydrates: _parseDouble(row[22]),
      category: row[7]?.toString() ?? '',
      subcategory: row[11]?.toString(),
    );
  }

  static double _parseDouble(dynamic value) {
    if (value == null || value.toString().isEmpty) return 0.0;
    try {
      return double.parse(value.toString());
    } catch (e) {
      return 0.0;
    }
  }

  @override
  String toString() {
    return 'FoodItem(name: $name, calories: $calories, '
        'protein: $protein, fat: $fat, carbs: $carbohydrates)';
  }
}

class FoodDatabaseService {
  static final FoodDatabaseService _instance = FoodDatabaseService._internal();
  factory FoodDatabaseService() => _instance;
  FoodDatabaseService._internal();

  List<FoodItem>? _foodDatabase;
  bool _isLoaded = false;

  // CSV 파일 로드 및 파싱
  Future<void> loadDatabase() async {
    if (_isLoaded) return;

    try {
      // assets에서 CSV 파일 읽기
      final csvString = await rootBundle.loadString('assets/food_database.csv');

      // CSV 파싱
      final List<List<dynamic>> csvTable = const CsvToListConverter().convert(
        csvString,
        eol: '\n',
      );

      // 헤더 제외하고 데이터만 파싱
      _foodDatabase = csvTable
          .skip(1) // 첫 번째 행(헤더) 제외
          .map((row) {
            try {
              return FoodItem.fromCsvRow(row);
            } catch (e) {
              return null;
            }
          })
          .where((item) => item != null)
          .cast<FoodItem>()
          .toList();

      _isLoaded = true;
      print('음식 데이터베이스 로드 완료: ${_foodDatabase?.length}개 항목');
    } catch (e) {
      print('음식 데이터베이스 로드 실패: $e');
      _foodDatabase = [];
      rethrow;
    }
  }

  // 데이터베이스가 로드되었는지 확인
  bool get isLoaded => _isLoaded;

  // 전체 데이터베이스 반환
  List<FoodItem> get allFoods {
    if (!_isLoaded) {
      throw Exception('데이터베이스가 아직 로드되지 않았습니다. loadDatabase()를 먼저 호출하세요.');
    }
    return _foodDatabase ?? [];
  }

  // 음식 이름으로 검색 (부분 일치)
  List<FoodItem> searchByName(String query) {
    if (!_isLoaded) return [];
    if (query.isEmpty) return [];

    final lowerQuery = query.toLowerCase().trim();
    return _foodDatabase!
        .where((food) => food.name.toLowerCase().contains(lowerQuery))
        .toList();
  }

  // 음식 이름으로 정확히 일치하는 항목 검색
  FoodItem? findExactMatch(String name) {
    if (!_isLoaded) return null;

    try {
      return _foodDatabase!.firstWhere(
        (food) => food.name.toLowerCase() == name.toLowerCase().trim(),
      );
    } catch (e) {
      return null;
    }
  }

  // 카테고리별 음식 검색
  List<FoodItem> searchByCategory(String category) {
    if (!_isLoaded) return [];

    return _foodDatabase!
        .where((food) => food.category.contains(category))
        .toList();
  }

  // 유사한 음식 찾기 (퍼지 매칭)
  List<FoodItem> findSimilarFoods(String name, {int limit = 10}) {
    if (!_isLoaded) return [];

    final query = name.toLowerCase().trim();
    final results = <MapEntry<FoodItem, int>>[];

    for (final food in _foodDatabase!) {
      final foodName = food.name.toLowerCase();

      // 완전 일치
      if (foodName == query) {
        results.add(MapEntry(food, 1000));
        continue;
      }

      // 포함 일치
      if (foodName.contains(query) || query.contains(foodName)) {
        results.add(MapEntry(food, 500));
        continue;
      }

      // 단어 단위 일치
      final queryWords = query.split(' ');
      final foodWords = foodName.split(' ');
      int matchCount = 0;

      for (final qWord in queryWords) {
        for (final fWord in foodWords) {
          if (qWord.isNotEmpty && fWord.contains(qWord)) {
            matchCount++;
          }
        }
      }

      if (matchCount > 0) {
        results.add(MapEntry(food, matchCount * 100));
      }
    }

    // 점수 순으로 정렬
    results.sort((a, b) => b.value.compareTo(a.value));

    // 상위 limit개만 반환
    return results.take(limit).map((e) => e.key).toList();
  }

  // 데이터베이스 통계
  Map<String, dynamic> getStatistics() {
    if (!_isLoaded) return {};

    return {
      'totalItems': _foodDatabase?.length ?? 0,
      'categories': _foodDatabase
              ?.map((food) => food.category)
              .toSet()
              .length ??
          0,
    };
  }
}
