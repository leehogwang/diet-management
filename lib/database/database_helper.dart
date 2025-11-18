import 'package:sqflite/sqflite.dart';
import 'package:path/path.dart';
import '../models/meal_nutrition.dart';

class DatabaseHelper {
  static final DatabaseHelper instance = DatabaseHelper._init();
  static Database? _database;

  DatabaseHelper._init();

  Future<Database> get database async {
    if (_database != null) return _database!;
    _database = await _initDB('meal_nutrition.db');
    return _database!;
  }

  Future<Database> _initDB(String filePath) async {
    final dbPath = await getDatabasesPath();
    final path = join(dbPath, filePath);

    return await openDatabase(
      path,
      version: 1,
      onCreate: _createDB,
    );
  }

  Future<void> _createDB(Database db, int version) async {
    const idType = 'INTEGER PRIMARY KEY AUTOINCREMENT';
    const textType = 'TEXT NOT NULL';
    const realType = 'REAL NOT NULL';

    await db.execute('''
      CREATE TABLE meal_nutrition (
        id $idType,
        dateTime $textType,
        mealType $textType,
        foodName $textType,
        calories $realType,
        carbohydrates $realType,
        protein $realType,
        fat $realType,
        notes TEXT
      )
    ''');
  }

  // 식단 데이터 추가
  Future<MealNutrition> create(MealNutrition meal) async {
    final db = await instance.database;
    final id = await db.insert('meal_nutrition', meal.toMap());
    return meal.copyWith(id: id);
  }

  // 특정 ID의 식단 데이터 조회
  Future<MealNutrition?> read(int id) async {
    final db = await instance.database;
    final maps = await db.query(
      'meal_nutrition',
      columns: [
        'id',
        'dateTime',
        'mealType',
        'foodName',
        'calories',
        'carbohydrates',
        'protein',
        'fat',
        'notes'
      ],
      where: 'id = ?',
      whereArgs: [id],
    );

    if (maps.isNotEmpty) {
      return MealNutrition.fromMap(maps.first);
    } else {
      return null;
    }
  }

  // 모든 식단 데이터 조회
  Future<List<MealNutrition>> readAll() async {
    final db = await instance.database;
    const orderBy = 'dateTime DESC';
    final result = await db.query('meal_nutrition', orderBy: orderBy);
    return result.map((json) => MealNutrition.fromMap(json)).toList();
  }

  // 특정 날짜의 식단 데이터 조회
  Future<List<MealNutrition>> readByDate(DateTime date) async {
    final db = await instance.database;
    final startOfDay = DateTime(date.year, date.month, date.day);
    final endOfDay = DateTime(date.year, date.month, date.day, 23, 59, 59);

    final result = await db.query(
      'meal_nutrition',
      where: 'dateTime >= ? AND dateTime <= ?',
      whereArgs: [
        startOfDay.toIso8601String(),
        endOfDay.toIso8601String(),
      ],
      orderBy: 'dateTime ASC',
    );

    return result.map((json) => MealNutrition.fromMap(json)).toList();
  }

  // 특정 날짜 범위의 식단 데이터 조회
  Future<List<MealNutrition>> readByDateRange(
      DateTime startDate, DateTime endDate) async {
    final db = await instance.database;

    final result = await db.query(
      'meal_nutrition',
      where: 'dateTime >= ? AND dateTime <= ?',
      whereArgs: [
        startDate.toIso8601String(),
        endDate.toIso8601String(),
      ],
      orderBy: 'dateTime ASC',
    );

    return result.map((json) => MealNutrition.fromMap(json)).toList();
  }

  // 식단 데이터 수정
  Future<int> update(MealNutrition meal) async {
    final db = await instance.database;
    return db.update(
      'meal_nutrition',
      meal.toMap(),
      where: 'id = ?',
      whereArgs: [meal.id],
    );
  }

  // 식단 데이터 삭제
  Future<int> delete(int id) async {
    final db = await instance.database;
    return await db.delete(
      'meal_nutrition',
      where: 'id = ?',
      whereArgs: [id],
    );
  }

  // 모든 데이터 삭제
  Future<int> deleteAll() async {
    final db = await instance.database;
    return await db.delete('meal_nutrition');
  }

  // 데이터베이스 닫기
  Future<void> close() async {
    final db = await instance.database;
    await db.close();
  }
}
