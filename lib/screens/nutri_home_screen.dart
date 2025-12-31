import 'package:flutter/material.dart';
import '../theme/app_theme.dart';
import '../theme/app_colors.dart';
import 'nutri_food_scanner_screen.dart';
import 'nutri_settings_screen.dart';
import 'nutrient_settings_modal.dart';
import 'personal_settings_screen.dart';

class NutriHomeScreen extends StatefulWidget {
  final String palette;
  final bool isDarkMode;

  const NutriHomeScreen({
    super.key,
    this.palette = 'sage',
    this.isDarkMode = false,
  });

  @override
  State<NutriHomeScreen> createState() => _NutriHomeScreenState();
}

class _NutriHomeScreenState extends State<NutriHomeScreen> {
  bool _isCalendarExpanded = false;
  int _selectedDate = 3;
  String _ringMetric = 'calories';
  List<String> _barMetrics = ['carbs', 'protein', 'sodium'];
  late String _currentPalette;
  late bool _currentIsDarkMode;
  int _activeTab = 0; // 0: home, 1: stats, 2: exercise, 3: profile

  // Daily Goals
  final Map<String, double> _dailyGoals = {
    'calories': 2000,
    'carbs': 250,
    'protein': 120,
    'fat': 65,
    'sugar': 50,
    'sodium': 2000,
    'fiber': 30,
  };

  // Daily Stats (Mock Data)
  final Map<String, double> _currentStats = {
    'calories': 1250,
    'carbs': 140,
    'protein': 85,
    'fat': 35,
    'sugar': 12,
    'sodium': 1200,
    'fiber': 15,
  };

  // Nutrient Meta
  final Map<String, NutrientMeta> _nutrientMeta = {
    'calories': NutrientMeta(label: '칼로리', unit: 'kcal'),
    'carbs': NutrientMeta(label: '탄수화물', unit: 'g'),
    'protein': NutrientMeta(label: '단백질', unit: 'g'),
    'fat': NutrientMeta(label: '지방', unit: 'g'),
    'sugar': NutrientMeta(label: '당류', unit: 'g'),
    'sodium': NutrientMeta(label: '나트륨', unit: 'mg'),
    'fiber': NutrientMeta(label: '식이섬유', unit: 'g'),
  };

  // Meal Data
  final List<MealData> _meals = [
    MealData(
      title: '아침',
      time: '08:30',
      calories: 450,
      items: ['된장찌개', '밥'],
      icon: '☀️',
    ),
    MealData(
      title: '점심',
      time: '12:45',
      calories: 620,
      items: ['제육볶음', '반찬'],
      icon: '🌤️',
    ),
  ];

  @override
  void initState() {
    super.initState();
    _currentPalette = widget.palette;
    _currentIsDarkMode = widget.isDarkMode;
  }

  @override
  Widget build(BuildContext context) {
    final theme = AppTheme.getTheme(_currentPalette, _currentIsDarkMode);
    final primaryColor = theme.colorScheme.primary;
    final cardColor = theme.colorScheme.surface;
    final textColor = theme.colorScheme.onSurface;
    final textGray = theme.colorScheme.onSurface.withOpacity(0.6);

    return Theme(
      data: theme,
      child: Scaffold(
        backgroundColor: theme.scaffoldBackgroundColor,
        body: SafeArea(
          child: _buildTabContent(theme, primaryColor, cardColor, textColor, textGray),
        ),
        floatingActionButton: Padding(
          padding: const EdgeInsets.only(bottom: 100),
          child: FloatingActionButton(
            onPressed: () {
              Navigator.push(
                context,
                MaterialPageRoute(
                  builder: (context) => NutriFoodScannerScreen(
                    palette: _currentPalette,
                    isDarkMode: _currentIsDarkMode,
                  ),
                ),
              );
            },
            backgroundColor: primaryColor,
            elevation: 8,
            child: Container(
              decoration: BoxDecoration(
                color: primaryColor,
                shape: BoxShape.circle,
                border: Border.all(
                  color: _currentIsDarkMode
                      ? const Color(0xFF111814)
                      : const Color(0xFFF2F2EB),
                  width: 6,
                ),
              ),
              child: const Icon(Icons.camera_alt, color: Colors.white, size: 28),
            ),
          ),
        ),
        floatingActionButtonLocation: FloatingActionButtonLocation.centerDocked,
        bottomNavigationBar: _buildBottomNav(theme, primaryColor, textGray),
      ),
    );
  }

  Widget _buildTabContent(ThemeData theme, Color primaryColor, Color cardColor,
      Color textColor, Color textGray) {
    switch (_activeTab) {
      case 0: // Home
        return SingleChildScrollView(
          child: Column(
            children: [
              // Header Section with Calendar
              _buildHeaderSection(theme, primaryColor, cardColor, textColor, textGray),

              // Content
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 24),
                child: Column(
                  children: [
                    const SizedBox(height: 16),

                    // Nutrient Overview Card
                    _buildNutrientOverview(theme, cardColor, textColor, textGray),

                    const SizedBox(height: 24),

                    // Meals Section
                    _buildMealsSection(theme, cardColor, textColor, textGray),

                    const SizedBox(height: 120), // Space for FAB and nav
                  ],
                ),
              ),
            ],
          ),
        );
      case 1: // Stats
        return _buildStatsScreen(theme, primaryColor, textColor, textGray);
      case 2: // Exercise
        return _buildExerciseScreen(theme, primaryColor, textColor, textGray);
      case 3: // Profile
        return _buildProfileScreen(theme, cardColor, primaryColor, textColor, textGray);
      default:
        return const SizedBox.shrink();
    }
  }

  Widget _buildStatsScreen(ThemeData theme, Color primaryColor, Color textColor, Color textGray) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Container(
              width: 100,
              height: 100,
              decoration: BoxDecoration(
                color: primaryColor.withOpacity(0.1),
                shape: BoxShape.circle,
              ),
              child: Icon(Icons.bar_chart_rounded, size: 48, color: primaryColor),
            ),
            const SizedBox(height: 24),
            Text(
              '통계',
              style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold, color: textColor),
            ),
            const SizedBox(height: 12),
            Text(
              '주간 리포트',
              style: TextStyle(fontSize: 16, color: textGray),
            ),
            const SizedBox(height: 32),
            Text(
              '통계 화면도 곧 업데이트 됩니다.',
              style: TextStyle(fontSize: 14, color: textGray),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildExerciseScreen(ThemeData theme, Color primaryColor, Color textColor, Color textGray) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Container(
              width: 100,
              height: 100,
              decoration: BoxDecoration(
                color: primaryColor.withOpacity(0.1),
                shape: BoxShape.circle,
              ),
              child: Icon(Icons.fitness_center_rounded, size: 48, color: primaryColor),
            ),
            const SizedBox(height: 24),
            Text(
              '운동',
              style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold, color: textColor),
            ),
            const SizedBox(height: 12),
            Text(
              '운동 기록 기능이 곧 추가될 예정입니다.',
              style: TextStyle(fontSize: 14, color: textGray),
              textAlign: TextAlign.center,
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildProfileScreen(ThemeData theme, Color cardColor, Color primaryColor,
      Color textColor, Color textGray) {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(24),
      child: Column(
        children: [
          const SizedBox(height: 32),

          // Profile Card
          Container(
            padding: const EdgeInsets.all(32),
            decoration: BoxDecoration(
              color: cardColor,
              borderRadius: BorderRadius.circular(32),
              boxShadow: [
                BoxShadow(
                  color: Colors.black.withOpacity(0.05),
                  blurRadius: 10,
                  offset: const Offset(0, 2),
                ),
              ],
            ),
            child: Column(
              children: [
                Container(
                  width: 100,
                  height: 100,
                  decoration: BoxDecoration(
                    color: primaryColor.withOpacity(0.1),
                    shape: BoxShape.circle,
                  ),
                  child: Icon(Icons.person_rounded, size: 48, color: primaryColor),
                ),
                const SizedBox(height: 16),
                Text(
                  '건강한 지민',
                  style: TextStyle(
                    fontSize: 24,
                    fontWeight: FontWeight.bold,
                    color: textColor,
                  ),
                ),
                const SizedBox(height: 8),
                Text(
                  '프로 멤버',
                  style: TextStyle(fontSize: 14, color: textGray),
                ),
              ],
            ),
          ),

          const SizedBox(height: 24),

          // Menu Items
          _buildProfileMenuItem(
            Icons.settings_rounded,
            '개인 설정',
            primaryColor,
            cardColor,
            textColor,
            textGray,
            onTap: () {
              Navigator.push(
                context,
                MaterialPageRoute(
                  builder: (context) => PersonalSettingsScreen(
                    palette: _currentPalette,
                    isDarkMode: _currentIsDarkMode,
                  ),
                ),
              );
            },
          ),

          const SizedBox(height: 12),

          _buildProfileMenuItem(
            Icons.favorite_rounded,
            '건강 데이터',
            primaryColor,
            cardColor,
            textColor,
            textGray,
          ),

          const SizedBox(height: 12),

          _buildProfileMenuItem(
            Icons.info_rounded,
            '고객센터',
            primaryColor,
            cardColor,
            textColor,
            textGray,
          ),

          const SizedBox(height: 120), // Space for nav
        ],
      ),
    );
  }

  Widget _buildProfileMenuItem(
    IconData icon,
    String label,
    Color primaryColor,
    Color cardColor,
    Color textColor,
    Color textGray, {
    VoidCallback? onTap,
  }) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.all(20),
        decoration: BoxDecoration(
          color: cardColor,
          borderRadius: BorderRadius.circular(24),
        ),
        child: Row(
          children: [
            Container(
              width: 40,
              height: 40,
              decoration: BoxDecoration(
                color: primaryColor.withOpacity(0.1),
                shape: BoxShape.circle,
              ),
              child: Icon(icon, color: primaryColor, size: 20),
            ),
            const SizedBox(width: 16),
            Expanded(
              child: Text(
                label,
                style: TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.w500,
                  color: textColor,
                ),
              ),
            ),
            Icon(Icons.chevron_right, color: textGray, size: 18),
          ],
        ),
      ),
    );
  }

  Widget _buildHeaderSection(ThemeData theme, Color primaryColor, Color cardColor,
      Color textColor, Color textGray) {
    return Container(
      decoration: BoxDecoration(
        color: primaryColor,
        borderRadius: const BorderRadius.only(
          bottomLeft: Radius.circular(40),
          bottomRight: Radius.circular(40),
        ),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.1),
            blurRadius: 20,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      padding: const EdgeInsets.fromLTRB(24, 48, 24, 24),
      child: Column(
        children: [
          // Header
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              GestureDetector(
                onTap: () {
                  setState(() {
                    _isCalendarExpanded = !_isCalendarExpanded;
                  });
                },
                child: Row(
                  children: [
                    Text(
                      _isCalendarExpanded ? '12월 식단' : '오늘의 섭취량',
                      style: const TextStyle(
                        fontSize: 28,
                        fontWeight: FontWeight.bold,
                        color: Colors.white,
                      ),
                    ),
                    const SizedBox(width: 8),
                    AnimatedRotation(
                      turns: _isCalendarExpanded ? 0.5 : 0,
                      duration: const Duration(milliseconds: 300),
                      child: const Icon(
                        Icons.keyboard_arrow_down,
                        color: Color(0xFFD4DCD6),
                        size: 20,
                      ),
                    ),
                  ],
                ),
              ),
              Row(
                children: [
                  IconButton(
                    icon: Icon(
                      _currentIsDarkMode ? Icons.light_mode : Icons.dark_mode,
                      color: Colors.white,
                    ),
                    onPressed: () {
                      setState(() {
                        _currentIsDarkMode = !_currentIsDarkMode;
                      });
                    },
                  ),
                  IconButton(
                    icon: const Icon(Icons.settings, color: Colors.white),
                    onPressed: () async {
                      final result = await Navigator.push(
                        context,
                        MaterialPageRoute(
                          builder: (context) => NutriSettingsScreen(
                            palette: _currentPalette,
                            isDarkMode: _currentIsDarkMode,
                          ),
                        ),
                      );
                      if (result != null && result is Map<String, dynamic>) {
                        setState(() {
                          if (result['palette'] != null) {
                            _currentPalette = result['palette'];
                          }
                          if (result['isDarkMode'] != null) {
                            _currentIsDarkMode = result['isDarkMode'];
                          }
                        });
                      }
                    },
                  ),
                ],
              ),
            ],
          ),
          const SizedBox(height: 16),

          // Date subtitle
          if (!_isCalendarExpanded)
            Align(
              alignment: Alignment.centerLeft,
              child: Text(
                '12월 ${_selectedDate}일 수요일',
                style: const TextStyle(
                  fontSize: 12,
                  color: Color(0xFFD4DCD6),
                  fontWeight: FontWeight.w500,
                  letterSpacing: 1.0,
                ),
              ),
            ),

          const SizedBox(height: 16),

          // Calendar / Date Selector
          _isCalendarExpanded
              ? _buildExpandedCalendar(theme, textColor, textGray)
              : _buildDateSelector(theme, textGray),
        ],
      ),
    );
  }

  Widget _buildDateSelector(ThemeData theme, Color textGray) {
    return SizedBox(
      height: 60,
      child: ListView.builder(
        scrollDirection: Axis.horizontal,
        itemCount: 7,
        itemBuilder: (context, index) {
          final days = ['일', '월', '화', '수', '목', '금', '토'];
          final date = 29 + index > 31 ? 29 + index - 31 : 29 + index;
          final isSelected = date == _selectedDate;
          final hasData = [1, 2, 3, 5, 8, 12, 15, 20].contains(date);

          return GestureDetector(
            onTap: () {
              setState(() {
                _selectedDate = date;
              });
            },
            child: Container(
              width: 50,
              margin: const EdgeInsets.only(right: 12),
              decoration: BoxDecoration(
                color: isSelected
                    ? theme.colorScheme.primary.withOpacity(0.15)
                    : Colors.transparent,
                borderRadius: BorderRadius.circular(16),
              ),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Text(
                    days[index],
                    style: TextStyle(
                      fontSize: 10,
                      fontWeight: FontWeight.bold,
                      letterSpacing: 1.5,
                      color: isSelected
                          ? theme.colorScheme.primary
                          : const Color(0xFFD4DCD6).withOpacity(0.6),
                    ),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    '$date',
                    style: TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                      color: isSelected
                          ? theme.colorScheme.primary
                          : const Color(0xFFD4DCD6),
                    ),
                  ),
                  if (hasData)
                    Container(
                      margin: const EdgeInsets.only(top: 4),
                      width: 6,
                      height: 6,
                      decoration: BoxDecoration(
                        shape: BoxShape.circle,
                        color: isSelected
                            ? theme.colorScheme.primary
                            : Colors.white.withOpacity(0.2),
                      ),
                    ),
                ],
              ),
            ),
          );
        },
      ),
    );
  }

  Widget _buildExpandedCalendar(ThemeData theme, Color textColor, Color textGray) {
    return Container(
      constraints: const BoxConstraints(maxHeight: 220),
      padding: const EdgeInsets.symmetric(vertical: 8),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          // Weekday headers
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceAround,
            children: ['일', '월', '화', '수', '목', '금', '토'].map((day) {
              return SizedBox(
                width: 36,
                height: 20,
                child: Center(
                  child: Text(
                    day,
                    style: TextStyle(
                      fontSize: 10,
                      fontWeight: FontWeight.bold,
                      letterSpacing: 1.2,
                      color: day == '일'
                          ? const Color(0xFFFF9B9B)
                          : Colors.white.withOpacity(0.4),
                    ),
                  ),
                ),
              );
            }).toList(),
          ),
          const SizedBox(height: 6),

          // Calendar grid - Fixed height instead of Expanded
          SizedBox(
            height: 160,
            child: GridView.builder(
              shrinkWrap: true,
              physics: const NeverScrollableScrollPhysics(),
              gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                crossAxisCount: 7,
                childAspectRatio: 1.1,
                crossAxisSpacing: 3,
                mainAxisSpacing: 3,
              ),
              itemCount: 35,
              itemBuilder: (context, index) {
                final day = index - 1;
                if (day <= 0 || day > 31) {
                  return const SizedBox.shrink();
                }
                final isSelected = day == _selectedDate;
                final hasData = [1, 2, 3, 5, 8, 12, 15, 20].contains(day);

                return GestureDetector(
                  onTap: () {
                    setState(() {
                      _selectedDate = day;
                    });
                  },
                  child: Container(
                    decoration: BoxDecoration(
                      color: isSelected
                          ? theme.colorScheme.primary.withOpacity(0.15)
                          : Colors.transparent,
                      borderRadius: BorderRadius.circular(10),
                    ),
                    child: Stack(
                      children: [
                        Center(
                          child: Text(
                            '$day',
                            style: TextStyle(
                              fontSize: 13,
                              fontWeight: isSelected ? FontWeight.bold : FontWeight.normal,
                              color: isSelected
                                  ? theme.colorScheme.primary
                                  : Colors.white.withOpacity(0.6),
                            ),
                          ),
                        ),
                        if (hasData)
                          Positioned(
                            bottom: 3,
                            left: 0,
                            right: 0,
                            child: Center(
                              child: Container(
                                width: 3,
                                height: 3,
                                decoration: BoxDecoration(
                                  shape: BoxShape.circle,
                                  color: isSelected
                                      ? theme.colorScheme.primary
                                      : Colors.white.withOpacity(0.4),
                                ),
                              ),
                            ),
                          ),
                      ],
                    ),
                  ),
                );
              },
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildNutrientOverview(ThemeData theme, Color cardColor,
      Color textColor, Color textGray) {
    return Container(
      padding: const EdgeInsets.all(24),
      decoration: BoxDecoration(
        color: cardColor,
        borderRadius: BorderRadius.circular(32),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.05),
            blurRadius: 20,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Header
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                '영양소 개요',
                style: TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                  color: textColor,
                ),
              ),
              IconButton(
                icon: Icon(Icons.more_horiz, color: textGray),
                onPressed: () async {
                  final result = await showDialog<Map<String, dynamic>>(
                    context: context,
                    builder: (context) => NutrientSettingsModal(
                      palette: _currentPalette,
                      isDarkMode: _currentIsDarkMode,
                      ringMetric: _ringMetric,
                      barMetrics: _barMetrics,
                    ),
                  );
                  if (result != null) {
                    setState(() {
                      if (result['ringMetric'] != null) {
                        _ringMetric = result['ringMetric'];
                      }
                      if (result['barMetrics'] != null) {
                        _barMetrics = List<String>.from(result['barMetrics']);
                      }
                    });
                  }
                },
                padding: EdgeInsets.zero,
                constraints: const BoxConstraints(),
              ),
            ],
          ),
          const SizedBox(height: 24),

          // Ring Chart and Progress Bars
          Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Ring Chart
              SizedBox(
                width: 128,
                height: 128,
                child: Stack(
                  children: [
                    // Background ring
                    SizedBox(
                      width: 128,
                      height: 128,
                      child: CircularProgressIndicator(
                        value: 1,
                        strokeWidth: 3,
                        backgroundColor: Colors.transparent,
                        valueColor: AlwaysStoppedAnimation<Color>(
                          _currentIsDarkMode
                              ? AppColors.sageDarkChartRing
                              : const Color(0xFFF2F2EB),
                        ),
                      ),
                    ),
                    // Progress ring
                    SizedBox(
                      width: 128,
                      height: 128,
                      child: CircularProgressIndicator(
                        value: (_currentStats[_ringMetric] ?? 0) /
                            (_dailyGoals[_ringMetric] ?? 1),
                        strokeWidth: 3,
                        backgroundColor: Colors.transparent,
                        valueColor: AlwaysStoppedAnimation<Color>(
                          AppColors.getNutrientColor(
                            _currentPalette,
                            _currentIsDarkMode,
                            _ringMetric,
                          ),
                        ),
                      ),
                    ),
                    // Center text
                    Center(
                      child: Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Text(
                            '${_currentStats[_ringMetric]?.toInt() ?? 0}',
                            style: TextStyle(
                              fontSize: 24,
                              fontWeight: FontWeight.bold,
                              color: textColor,
                            ),
                          ),
                          Text(
                            _nutrientMeta[_ringMetric]?.unit ?? '',
                            style: TextStyle(
                              fontSize: 10,
                              color: textGray,
                              letterSpacing: 1.5,
                            ),
                          ),
                          const SizedBox(height: 4),
                          Text(
                            _nutrientMeta[_ringMetric]?.label ?? '',
                            style: TextStyle(
                              fontSize: 10,
                              color: textGray,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(width: 32),

              // Progress Bars
              Expanded(
                child: Column(
                  children: _barMetrics.map((metric) {
                    return _buildProgressBar(
                      metric,
                      _currentStats[metric] ?? 0,
                      _dailyGoals[metric] ?? 1,
                      textColor,
                      textGray,
                    );
                  }).toList(),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildProgressBar(String metric, double current, double max,
      Color textColor, Color textGray) {
    final meta = _nutrientMeta[metric];
    final percentage = (current / max).clamp(0.0, 1.0);
    final color = AppColors.getNutrientColor(
      _currentPalette,
      _currentIsDarkMode,
      metric,
    );

    return Padding(
      padding: const EdgeInsets.only(bottom: 16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                meta?.label ?? '',
                style: TextStyle(
                  fontSize: 12,
                  fontWeight: FontWeight.w500,
                  color: textGray,
                  letterSpacing: 0.5,
                ),
              ),
              Text(
                '${current.toInt()}/${max.toInt()}${meta?.unit ?? ''}',
                style: TextStyle(
                  fontSize: 12,
                  fontWeight: FontWeight.bold,
                  color: textColor,
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Container(
            height: 8,
            decoration: BoxDecoration(
              color: _currentIsDarkMode
                  ? Colors.white.withOpacity(0.1)
                  : const Color(0xFFF2F2EB),
              borderRadius: BorderRadius.circular(4),
            ),
            child: FractionallySizedBox(
              alignment: Alignment.centerLeft,
              widthFactor: percentage,
              child: Container(
                decoration: BoxDecoration(
                  color: color,
                  borderRadius: BorderRadius.circular(4),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildMealsSection(ThemeData theme, Color cardColor,
      Color textColor, Color textGray) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // Header
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 8),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                _isCalendarExpanded
                    ? '${_selectedDate}일의 식사'
                    : '오늘의 식사',
                style: TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                  color: textColor,
                ),
              ),
              GestureDetector(
                onTap: () {},
                child: Text(
                  '전체보기',
                  style: TextStyle(
                    fontSize: 12,
                    fontWeight: FontWeight.w500,
                    color: textGray,
                  ),
                ),
              ),
            ],
          ),
        ),
        const SizedBox(height: 16),

        // Meal Cards
        ..._meals.map((meal) {
          final breakfastColor = _currentIsDarkMode
              ? (_currentPalette == 'sage'
                  ? AppColors.sageDarkBreakfastBg
                  : _currentPalette == 'berry'
                      ? AppColors.berryDarkBreakfastBg
                      : AppColors.midnightDarkBreakfastBg)
              : (_currentPalette == 'sage'
                  ? AppColors.sageBreakfastBg
                  : _currentPalette == 'berry'
                      ? AppColors.berryBreakfastBg
                      : AppColors.midnightBreakfastBg);

          final lunchColor = _currentIsDarkMode
              ? (_currentPalette == 'sage'
                  ? AppColors.sageDarkLunchBg
                  : _currentPalette == 'berry'
                      ? AppColors.berryDarkLunchBg
                      : AppColors.midnightDarkLunchBg)
              : (_currentPalette == 'sage'
                  ? AppColors.sageLunchBg
                  : _currentPalette == 'berry'
                      ? AppColors.berryLunchBg
                      : AppColors.midnightLunchBg);

          final iconColor = meal.title == '아침'
              ? (_currentIsDarkMode
                  ? (_currentPalette == 'sage'
                      ? AppColors.sageDarkBreakfastIcon
                      : _currentPalette == 'berry'
                          ? AppColors.berryDarkBreakfastIcon
                          : AppColors.midnightDarkBreakfastIcon)
                  : (_currentPalette == 'sage'
                      ? AppColors.sageBreakfastIcon
                      : _currentPalette == 'berry'
                          ? AppColors.berryBreakfastIcon
                          : AppColors.midnightBreakfastIcon))
              : (_currentIsDarkMode
                  ? (_currentPalette == 'sage'
                      ? AppColors.sageDarkLunchIcon
                      : _currentPalette == 'berry'
                          ? AppColors.berryDarkLunchIcon
                          : AppColors.midnightDarkLunchIcon)
                  : (_currentPalette == 'sage'
                      ? AppColors.sageLunchIcon
                      : _currentPalette == 'berry'
                          ? AppColors.berryLunchIcon
                          : AppColors.midnightLunchIcon));

          final bgColor = meal.title == '아침' ? breakfastColor : lunchColor;

          return GestureDetector(
            onTap: () {},
            child: Container(
              margin: const EdgeInsets.only(bottom: 16),
              decoration: BoxDecoration(
                color: cardColor,
                borderRadius: BorderRadius.circular(28),
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withOpacity(0.05),
                    blurRadius: 10,
                    offset: const Offset(0, 2),
                  ),
                ],
              ),
              clipBehavior: Clip.antiAlias,
              child: Row(
                children: [
                  // Icon container
                  Container(
                    width: 80,
                    height: 100,
                    decoration: BoxDecoration(color: bgColor),
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Text(
                          meal.icon,
                          style: const TextStyle(fontSize: 24),
                        ),
                        const SizedBox(height: 4),
                        Text(
                          '${meal.calories}',
                          style: TextStyle(
                            fontSize: 16,
                            fontWeight: FontWeight.bold,
                            color: iconColor,
                          ),
                        ),
                        Text(
                          'Kcal',
                          style: TextStyle(
                            fontSize: 8,
                            fontWeight: FontWeight.bold,
                            color: iconColor.withOpacity(0.7),
                            letterSpacing: 1.0,
                          ),
                        ),
                      ],
                    ),
                  ),
                  // Info
                  Expanded(
                    child: Padding(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 20,
                        vertical: 20,
                      ),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Row(
                            mainAxisAlignment: MainAxisAlignment.spaceBetween,
                            children: [
                              Text(
                                meal.title,
                                style: TextStyle(
                                  fontSize: 18,
                                  fontWeight: FontWeight.bold,
                                  color: textColor,
                                ),
                              ),
                              Text(
                                meal.time,
                                style: TextStyle(
                                  fontSize: 12,
                                  color: textGray,
                                ),
                              ),
                            ],
                          ),
                          const SizedBox(height: 4),
                          Text(
                            meal.items.join(', '),
                            style: TextStyle(
                              fontSize: 14,
                              fontWeight: FontWeight.w500,
                              color: textGray,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                ],
              ),
            ),
          );
        }),
      ],
    );
  }

  Widget _buildBottomNav(ThemeData theme, Color primaryColor, Color textGray) {
    return Container(
      margin: const EdgeInsets.all(24),
      decoration: BoxDecoration(
        color: _currentIsDarkMode
            ? AppColors.sageDarkCardBg.withOpacity(0.95)
            : Colors.white.withOpacity(0.9),
        borderRadius: BorderRadius.circular(32),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.04),
            blurRadius: 30,
            offset: const Offset(0, 8),
          ),
        ],
      ),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 16),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            _buildNavItem(Icons.restaurant_rounded, _activeTab == 0, primaryColor, textGray, onTap: () => setState(() => _activeTab = 0)),
            _buildNavItem(Icons.bar_chart_rounded, _activeTab == 1, primaryColor, textGray, onTap: () => setState(() => _activeTab = 1)),
            const SizedBox(width: 32), // Space for FAB
            _buildNavItem(Icons.fitness_center_rounded, _activeTab == 2, primaryColor, textGray, onTap: () => setState(() => _activeTab = 2)),
            _buildNavItem(Icons.person_rounded, _activeTab == 3, primaryColor, textGray, onTap: () => setState(() => _activeTab = 3)),
          ],
        ),
      ),
    );
  }

  Widget _buildNavItem(IconData icon, bool isActive, Color primaryColor, Color textGray, {VoidCallback? onTap}) {
    return GestureDetector(
      onTap: onTap,
      child: Icon(
        icon,
        color: isActive ? primaryColor : textGray,
        size: 24,
      ),
    );
  }
}

// Data Models
class NutrientMeta {
  final String label;
  final String unit;

  NutrientMeta({required this.label, required this.unit});
}

class MealData {
  final String title;
  final String time;
  final int calories;
  final List<String> items;
  final String icon;

  MealData({
    required this.title,
    required this.time,
    required this.calories,
    required this.items,
    required this.icon,
  });
}
