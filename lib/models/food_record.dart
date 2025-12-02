class FoodRecord {
  final String id;
  final String foodName;
  final String mealType; // breakfast, lunch, dinner, snack
  final DateTime timestamp;
  final int calories;
  final double protein; // g
  final double carbs; // g
  final double fat; // g
  final double sodium; // mg
  final double sugar; // g
  final String? imageUrl;

  FoodRecord({
    required this.id,
    required this.foodName,
    required this.mealType,
    required this.timestamp,
    required this.calories,
    required this.protein,
    required this.carbs,
    required this.fat,
    required this.sodium,
    required this.sugar,
    this.imageUrl,
  });

  factory FoodRecord.fromJson(Map<String, dynamic> json) {
    return FoodRecord(
      id: json['id'],
      foodName: json['foodName'],
      mealType: json['mealType'],
      timestamp: DateTime.parse(json['timestamp']),
      calories: json['calories'],
      protein: json['protein'].toDouble(),
      carbs: json['carbs'].toDouble(),
      fat: json['fat'].toDouble(),
      sodium: json['sodium'].toDouble(),
      sugar: json['sugar'].toDouble(),
      imageUrl: json['imageUrl'],
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'foodName': foodName,
      'mealType': mealType,
      'timestamp': timestamp.toIso8601String(),
      'calories': calories,
      'protein': protein,
      'carbs': carbs,
      'fat': fat,
      'sodium': sodium,
      'sugar': sugar,
      'imageUrl': imageUrl,
    };
  }
}

// 더미 데이터
class DummyFoodData {
  static List<FoodRecord> getTodayRecords() {
    final now = DateTime.now();
    final today = DateTime(now.year, now.month, now.day);

    return [
      FoodRecord(
        id: '1',
        foodName: '김치찌개',
        mealType: 'breakfast',
        timestamp: today.add(const Duration(hours: 8)),
        calories: 450,
        protein: 25.0,
        carbs: 35.0,
        fat: 20.0,
        sodium: 1200.0,
        sugar: 5.0,
      ),
      FoodRecord(
        id: '2',
        foodName: '흰쌀밥',
        mealType: 'breakfast',
        timestamp: today.add(const Duration(hours: 8, minutes: 5)),
        calories: 300,
        protein: 5.0,
        carbs: 65.0,
        fat: 1.0,
        sodium: 0.0,
        sugar: 0.5,
      ),
      FoodRecord(
        id: '3',
        foodName: '라면',
        mealType: 'lunch',
        timestamp: today.add(const Duration(hours: 13)),
        calories: 510,
        protein: 10.0,
        carbs: 80.0,
        fat: 16.0,
        sodium: 1890.0,
        sugar: 8.0,
      ),
      FoodRecord(
        id: '4',
        foodName: '떡볶이',
        mealType: 'lunch',
        timestamp: today.add(const Duration(hours: 13, minutes: 10)),
        calories: 380,
        protein: 8.0,
        carbs: 70.0,
        fat: 7.0,
        sodium: 950.0,
        sugar: 25.0,
      ),
      FoodRecord(
        id: '5',
        foodName: '치킨',
        mealType: 'dinner',
        timestamp: today.add(const Duration(hours: 19)),
        calories: 800,
        protein: 45.0,
        carbs: 40.0,
        fat: 50.0,
        sodium: 1500.0,
        sugar: 10.0,
      ),
      FoodRecord(
        id: '6',
        foodName: '콜라',
        mealType: 'dinner',
        timestamp: today.add(const Duration(hours: 19, minutes: 20)),
        calories: 150,
        protein: 0.0,
        carbs: 39.0,
        fat: 0.0,
        sodium: 45.0,
        sugar: 39.0,
      ),
    ];
  }

  static List<FoodRecord> getYesterdayRecords() {
    final now = DateTime.now();
    final yesterday = DateTime(now.year, now.month, now.day - 1);

    return [
      FoodRecord(
        id: '7',
        foodName: '샐러드',
        mealType: 'breakfast',
        timestamp: yesterday.add(const Duration(hours: 8)),
        calories: 250,
        protein: 12.0,
        carbs: 30.0,
        fat: 8.0,
        sodium: 350.0,
        sugar: 8.0,
      ),
      FoodRecord(
        id: '8',
        foodName: '삼겹살',
        mealType: 'dinner',
        timestamp: yesterday.add(const Duration(hours: 19)),
        calories: 700,
        protein: 35.0,
        carbs: 5.0,
        fat: 60.0,
        sodium: 1100.0,
        sugar: 2.0,
      ),
    ];
  }

  static List<FoodRecord> getAllRecords() {
    return [...getTodayRecords(), ...getYesterdayRecords()];
  }
}

// 일일 영양 요약
class DailyNutritionSummary {
  final int totalCalories;
  final double totalProtein;
  final double totalCarbs;
  final double totalFat;
  final double totalSodium;
  final double totalSugar;

  // 권장 섭취량
  static const int recommendedCalories = 2000;
  static const double recommendedProtein = 55.0;
  static const double recommendedCarbs = 300.0;
  static const double recommendedFat = 55.0;
  static const double recommendedSodium = 2000.0;
  static const double recommendedSugar = 50.0;

  DailyNutritionSummary({
    required this.totalCalories,
    required this.totalProtein,
    required this.totalCarbs,
    required this.totalFat,
    required this.totalSodium,
    required this.totalSugar,
  });

  factory DailyNutritionSummary.fromRecords(List<FoodRecord> records) {
    return DailyNutritionSummary(
      totalCalories: records.fold(0, (sum, r) => sum + r.calories),
      totalProtein: records.fold(0.0, (sum, r) => sum + r.protein),
      totalCarbs: records.fold(0.0, (sum, r) => sum + r.carbs),
      totalFat: records.fold(0.0, (sum, r) => sum + r.fat),
      totalSodium: records.fold(0.0, (sum, r) => sum + r.sodium),
      totalSugar: records.fold(0.0, (sum, r) => sum + r.sugar),
    );
  }

  bool isCaloriesExceeded() => totalCalories > recommendedCalories;
  bool isProteinExceeded() => totalProtein > recommendedProtein;
  bool isCarbsExceeded() => totalCarbs > recommendedCarbs;
  bool isFatExceeded() => totalFat > recommendedFat;
  bool isSodiumExceeded() => totalSodium > recommendedSodium;
  bool isSugarExceeded() => totalSugar > recommendedSugar;

  double getCaloriesPercentage() => (totalCalories / recommendedCalories) * 100;
  double getProteinPercentage() => (totalProtein / recommendedProtein) * 100;
  double getCarbsPercentage() => (totalCarbs / recommendedCarbs) * 100;
  double getFatPercentage() => (totalFat / recommendedFat) * 100;
  double getSodiumPercentage() => (totalSodium / recommendedSodium) * 100;
  double getSugarPercentage() => (totalSugar / recommendedSugar) * 100;
}
