class MealNutrition {
  final int? id;
  final DateTime dateTime;
  final String mealType; // 아침, 점심, 저녁, 간식
  final String foodName;
  final double calories; // 칼로리 (kcal)
  final double carbohydrates; // 탄수화물 (g)
  final double protein; // 단백질 (g)
  final double fat; // 지방 (g)
  final String? notes; // 메모

  MealNutrition({
    this.id,
    required this.dateTime,
    required this.mealType,
    required this.foodName,
    required this.calories,
    required this.carbohydrates,
    required this.protein,
    required this.fat,
    this.notes,
  });

  // 데이터베이스에서 Map으로부터 객체 생성
  factory MealNutrition.fromMap(Map<String, dynamic> map) {
    return MealNutrition(
      id: map['id'] as int?,
      dateTime: DateTime.parse(map['dateTime'] as String),
      mealType: map['mealType'] as String,
      foodName: map['foodName'] as String,
      calories: (map['calories'] as num).toDouble(),
      carbohydrates: (map['carbohydrates'] as num).toDouble(),
      protein: (map['protein'] as num).toDouble(),
      fat: (map['fat'] as num).toDouble(),
      notes: map['notes'] as String?,
    );
  }

  // 객체를 Map으로 변환 (데이터베이스 저장용)
  Map<String, dynamic> toMap() {
    return {
      'id': id,
      'dateTime': dateTime.toIso8601String(),
      'mealType': mealType,
      'foodName': foodName,
      'calories': calories,
      'carbohydrates': carbohydrates,
      'protein': protein,
      'fat': fat,
      'notes': notes,
    };
  }

  // 복사본 생성 (수정 시 사용)
  MealNutrition copyWith({
    int? id,
    DateTime? dateTime,
    String? mealType,
    String? foodName,
    double? calories,
    double? carbohydrates,
    double? protein,
    double? fat,
    String? notes,
  }) {
    return MealNutrition(
      id: id ?? this.id,
      dateTime: dateTime ?? this.dateTime,
      mealType: mealType ?? this.mealType,
      foodName: foodName ?? this.foodName,
      calories: calories ?? this.calories,
      carbohydrates: carbohydrates ?? this.carbohydrates,
      protein: protein ?? this.protein,
      fat: fat ?? this.fat,
      notes: notes ?? this.notes,
    );
  }

  @override
  String toString() {
    return 'MealNutrition(id: $id, dateTime: $dateTime, mealType: $mealType, '
        'foodName: $foodName, calories: $calories, carbs: $carbohydrates, '
        'protein: $protein, fat: $fat, notes: $notes)';
  }
}
