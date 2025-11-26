import 'package:flutter/material.dart';
import 'package:flutter/foundation.dart';
import 'package:camera/camera.dart';
import 'dart:io';
import 'dart:convert';
import 'dart:async';
import 'dart:typed_data';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as path;
import 'package:image_picker/image_picker.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'package:fl_chart/fl_chart.dart';
import '../services/openai_service.dart';
import '../models/nutrition_data.dart';
import 'nutrition_monitoring_screen.dart';

class MainNavigationScreen extends StatefulWidget {
  const MainNavigationScreen({super.key});

  @override
  State<MainNavigationScreen> createState() => _MainNavigationScreenState();
}

class _MainNavigationScreenState extends State<MainNavigationScreen> {
  int _currentIndex = 0;

  late final List<Widget> _pages;

  @override
  void initState() {
    super.initState();
    _pages = [
      HomePage(onNavigateToNutrition: () => _navigateToTab(2)),
      const CameraPage(),
      const NutritionMonitoringScreen(key: ValueKey('nutrition')),
      const UserInfoPage(),
    ];
  }

  void _navigateToTab(int index) {
    setState(() {
      _currentIndex = index;
    });
  }

  void _refreshNutritionData() {
    // NutritionMonitoringScreen이 새로고갈되도록 키 변경
    setState(() {
      _pages[2] = NutritionMonitoringScreen(key: ValueKey('nutrition_${DateTime.now().millisecondsSinceEpoch}'));
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: _pages[_currentIndex],
      bottomNavigationBar: BottomNavigationBar(
        type: BottomNavigationBarType.fixed,
        currentIndex: _currentIndex,
        onTap: (index) {
          setState(() {
            _currentIndex = index;
          });

          // Nutrition 탭으로 이동 시 데이터 새로고침
          if (index == 2) {
            _refreshNutritionData();
          }
        },
        items: const [
          BottomNavigationBarItem(
            icon: Icon(Icons.home),
            label: 'Home',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.camera_alt),
            label: 'Photo',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.bar_chart),
            label: 'Nutrition',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.person),
            label: 'User Infor',
          ),
        ],
      ),
    );
  }
}

class HomePage extends StatefulWidget {
  final VoidCallback onNavigateToNutrition;

  const HomePage({super.key, required this.onNavigateToNutrition});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  DateTime _selectedDate = DateTime.now();

  void _showPlaceholderDialog(String title) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Text(title),
        content: const Text('준비 중입니다.'),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('확인'),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Home'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        actions: [
          IconButton(
            icon: const Icon(Icons.notifications),
            onPressed: () => _showPlaceholderDialog('공지사항'),
          ),
        ],
      ),
      drawer: Drawer(
        child: ListView(
          padding: EdgeInsets.zero,
          children: [
            const DrawerHeader(
              decoration: BoxDecoration(
                color: Colors.deepOrange,
              ),
              child: Text(
                '메뉴',
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 24,
                ),
              ),
            ),
            ListTile(
              leading: const Icon(Icons.circle_outlined),
              title: const Text('Patch-1 Features'),
              onTap: () {
                Navigator.pop(context);
                _showPlaceholderDialog('Patch-1 Features');
              },
            ),
            ListTile(
              leading: const Icon(Icons.circle_outlined),
              title: const Text('Yoon-1 Features'),
              onTap: () {
                Navigator.pop(context);
                _showPlaceholderDialog('Yoon-1 Features');
              },
            ),
            ListTile(
              leading: const Icon(Icons.circle_outlined),
              title: const Text('Data Import'),
              onTap: () {
                Navigator.pop(context);
                _showPlaceholderDialog('Data Import');
              },
            ),
            ListTile(
              leading: const Icon(Icons.help_outline),
              title: const Text('ask'),
              onTap: () {
                Navigator.pop(context);
                _showPlaceholderDialog('문의하기');
              },
            ),
            ListTile(
              leading: const Icon(Icons.settings),
              title: const Text('Settings'),
              onTap: () {
                Navigator.pop(context);
                _showPlaceholderDialog('설정');
              },
            ),
          ],
        ),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Profile and User Info in Row
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              crossAxisAlignment: CrossAxisAlignment.center,
              children: [
                // Profile Section (Left) - Just circle, no card
                InkWell(
                  onTap: () => _showPlaceholderDialog('Profile'),
                  child: const Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      CircleAvatar(
                        radius: 35,
                        child: Icon(Icons.person, size: 40),
                      ),
                      SizedBox(height: 8),
                      Text(
                        'Profile',
                        style: TextStyle(
                          fontSize: 12,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ],
                  ),
                ),
                const SizedBox(width: 16),

                // Simple User Information Section (Right)
                Expanded(
                  child: Card(
                    elevation: 2,
                    child: InkWell(
                      onTap: () => _showPlaceholderDialog('Simple User Informations'),
                      child: const Padding(
                        padding: EdgeInsets.all(16.0),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Row(
                              children: [
                                Icon(Icons.info_outline, size: 30),
                                SizedBox(width: 12),
                                Column(
                                  crossAxisAlignment: CrossAxisAlignment.start,
                                  children: [
                                    Text(
                                      'Simple User',
                                      style: TextStyle(
                                        fontSize: 16,
                                        fontWeight: FontWeight.bold,
                                      ),
                                    ),
                                    Text(
                                      'Informations',
                                      style: TextStyle(
                                        fontSize: 16,
                                        fontWeight: FontWeight.bold,
                                      ),
                                    ),
                                  ],
                                ),
                              ],
                            ),
                            SizedBox(height: 8),
                            Text(
                              '이름, 기본 정보',
                              style: TextStyle(fontSize: 12),
                            ),
                          ],
                        ),
                      ),
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 20),

            // Calendar Section
            const Text(
              '캘린더',
              style: TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 12),
            Card(
              elevation: 4,
              child: Padding(
                padding: const EdgeInsets.all(16.0),
                child: Column(
                  children: [
                    // Month/Year Header
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        IconButton(
                          icon: const Icon(Icons.chevron_left),
                          onPressed: () {
                            setState(() {
                              _selectedDate = DateTime(
                                _selectedDate.year,
                                _selectedDate.month - 1,
                              );
                            });
                          },
                        ),
                        Text(
                          '${_selectedDate.year}년 ${_selectedDate.month}월',
                          style: const TextStyle(
                            fontSize: 18,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        IconButton(
                          icon: const Icon(Icons.chevron_right),
                          onPressed: () {
                            setState(() {
                              _selectedDate = DateTime(
                                _selectedDate.year,
                                _selectedDate.month + 1,
                              );
                            });
                          },
                        ),
                      ],
                    ),
                    const SizedBox(height: 16),

                    // Calendar Grid
                    _buildCalendar(),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildCalendar() {
    final firstDayOfMonth = DateTime(_selectedDate.year, _selectedDate.month, 1);
    final lastDayOfMonth = DateTime(_selectedDate.year, _selectedDate.month + 1, 0);
    final daysInMonth = lastDayOfMonth.day;
    final startWeekday = firstDayOfMonth.weekday % 7; // 0 = Sunday

    return Column(
      children: [
        // Weekday headers
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceAround,
          children: ['일', '월', '화', '수', '목', '금', '토']
              .map((day) => Expanded(
                    child: Center(
                      child: Text(
                        day,
                        style: TextStyle(
                          fontWeight: FontWeight.bold,
                          color: day == '일'
                              ? Colors.red
                              : day == '토'
                                  ? Colors.blue
                                  : Colors.black,
                        ),
                      ),
                    ),
                  ))
              .toList(),
        ),
        const SizedBox(height: 8),

        // Calendar days
        ...List.generate((daysInMonth + startWeekday) ~/ 7 + 1, (weekIndex) {
          return Padding(
            padding: const EdgeInsets.symmetric(vertical: 4.0),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceAround,
              children: List.generate(7, (dayIndex) {
                final dayNumber = weekIndex * 7 + dayIndex - startWeekday + 1;

                if (dayNumber < 1 || dayNumber > daysInMonth) {
                  return const Expanded(child: SizedBox());
                }

                final isToday = dayNumber == DateTime.now().day &&
                    _selectedDate.month == DateTime.now().month &&
                    _selectedDate.year == DateTime.now().year;

                return Expanded(
                  child: Container(
                    height: 60,
                    margin: const EdgeInsets.all(2),
                    decoration: BoxDecoration(
                      color: isToday ? Colors.orange.shade100 : null,
                      borderRadius: BorderRadius.circular(8),
                      border: Border.all(
                        color: Colors.grey.shade300,
                        width: 0.5,
                      ),
                    ),
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.start,
                      children: [
                        const SizedBox(height: 4),
                        Text(
                          '$dayNumber',
                          style: TextStyle(
                            color: dayIndex == 0
                                ? Colors.red
                                : dayIndex == 6
                                    ? Colors.blue
                                    : Colors.black,
                            fontWeight: isToday ? FontWeight.bold : FontWeight.normal,
                            fontSize: 14,
                          ),
                        ),
                        const SizedBox(height: 4),
                        // Blank space for future information - 클릭 시 Nutrition 탭으로 이동
                        InkWell(
                          onTap: () {
                            widget.onNavigateToNutrition();
                          },
                          child: Container(
                            width: 30,
                            height: 20,
                            decoration: BoxDecoration(
                              color: Colors.grey.shade100,
                              borderRadius: BorderRadius.circular(4),
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                );
              }),
            ),
          );
        }),
      ],
    );
  }
}

class UserInfoPage extends StatelessWidget {
  const UserInfoPage({super.key});

  void _showPlaceholderDialog(BuildContext context, String title) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Text(title),
        content: Text('$title 기능은 준비 중입니다.'),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('확인'),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('User Information'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Card(
              elevation: 4,
              child: InkWell(
                onTap: () => _showPlaceholderDialog(context, 'Profile'),
                child: const Padding(
                  padding: EdgeInsets.all(16.0),
                  child: Row(
                    children: [
                      Icon(Icons.person, size: 40),
                      SizedBox(width: 16),
                      Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            'Profile',
                            style: TextStyle(
                              fontSize: 20,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                          SizedBox(height: 4),
                          Text('프로필 정보 보기'),
                        ],
                      ),
                      Spacer(),
                      Icon(Icons.arrow_forward_ios),
                    ],
                  ),
                ),
              ),
            ),
            const SizedBox(height: 16),
            Card(
              elevation: 4,
              child: InkWell(
                onTap: () => _showPlaceholderDialog(context, 'Simple User Informations'),
                child: const Padding(
                  padding: EdgeInsets.all(16.0),
                  child: Row(
                    children: [
                      Icon(Icons.info, size: 40),
                      SizedBox(width: 16),
                      Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            'Simple User Informations',
                            style: TextStyle(
                              fontSize: 20,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                          SizedBox(height: 4),
                          Text('간단한 사용자 정보'),
                        ],
                      ),
                      Spacer(),
                      Icon(Icons.arrow_forward_ios),
                    ],
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

// Camera Page - Reusing your existing camera functionality
class CameraPage extends StatefulWidget {
  const CameraPage({super.key});

  @override
  State<CameraPage> createState() => _CameraPageState();
}

class _CameraPageState extends State<CameraPage> {
  CameraController? _cameraController;
  List<CameraDescription>? _cameras;
  bool _isInitialized = false;
  bool _isLoading = false;
  ClaudeService? _claudeService;

  // Draggable button state
  Offset _buttonPosition = Offset.zero;
  bool _isPositioned = false;
  bool _isLongPressing = false;
  Timer? _resetTimer;
  Offset? _panStartPosition;
  bool _hasMoved = false;
  bool _resetTimerExecuted = false;
  bool _isResetting = false;
  Offset _resetStartPosition = Offset.zero;
  Offset _homePosition = Offset.zero;

  // Zoom state
  double _currentZoom = 1.0;
  double _minZoom = 1.0;
  double _maxZoom = 1.0;
  bool _showZoomSlider = false;
  Timer? _zoomSliderTimer;
  double _baseZoom = 1.0;

  // Pinch gesture tracking
  double? _initialDistance;
  Map<int, Offset> _pointers = {};

  @override
  void initState() {
    super.initState();
    _initializeCamera();
    final apiKey = dotenv.env['CLAUDE_API_KEY'];
    if (apiKey == null || apiKey.isEmpty) {
      debugPrint('❌ API KEY is null or empty!');
      debugPrint('Available env keys: ${dotenv.env.keys.toList()}');
    } else {
      debugPrint('✅ Claude API KEY loaded: ${apiKey.substring(0, 20)}...');
    }
    _claudeService = ClaudeService(apiKey ?? '');
  }

  Future<void> _initializeCamera() async {
    try {
      _cameras = await availableCameras();
      if (_cameras != null && _cameras!.isNotEmpty) {
        _cameraController = CameraController(
          _cameras![0],
          ResolutionPreset.high,
          enableAudio: false,
        );
        await _cameraController!.initialize();

        // Set focus mode to auto
        try {
          await _cameraController!.setFocusMode(FocusMode.auto);
        } catch (e) {
          debugPrint('Focus mode not supported: $e');
        }

        // Get zoom capabilities
        try {
          _minZoom = await _cameraController!.getMinZoomLevel();
          _maxZoom = await _cameraController!.getMaxZoomLevel();
          _currentZoom = 1.0;
          _baseZoom = 1.0;

          debugPrint('Device zoom range: $_minZoom - $_maxZoom');
        } catch (e) {
          debugPrint('Zoom not supported: $e');
          _minZoom = 1.0;
          _maxZoom = 1.0;
        }

        if (mounted) {
          setState(() {
            _isInitialized = true;
          });
        }
      }
    } catch (e) {
      debugPrint('Error initializing camera: $e');
    }
  }

  @override
  void dispose() {
    _resetTimer?.cancel();
    _zoomSliderTimer?.cancel();
    _cameraController?.dispose();
    super.dispose();
  }

  Future<void> _takePicture() async {
    if (_cameraController == null || !_cameraController!.value.isInitialized) {
      return;
    }

    setState(() {
      _isLoading = true;
    });

    try {
      final XFile photo = await _cameraController!.takePicture();

      if (mounted) {
        final directory = await getApplicationDocumentsDirectory();
        final foodPhotosDir = Directory('${directory.path}/temp_food_photos');
        if (!await foodPhotosDir.exists()) {
          await foodPhotosDir.create(recursive: true);
        }

        final timestamp = DateTime.now().millisecondsSinceEpoch;
        final savedPath = '${foodPhotosDir.path}/food_$timestamp.jpg';
        await File(photo.path).copy(savedPath);

        // AI analysis
        try {
          if (_claudeService == null) {
            throw Exception('Claude 서비스가 초기화되지 않았습니다');
          }
          final analysisResult = await _claudeService!.analyzeFoodImage(File(savedPath));

          final foods = analysisResult['foods'] as List<dynamic>? ?? [];
          final markers = foods.map((food) => {
            'x': food['x'] ?? 0.5,
            'y': food['y'] ?? 0.5,
            'label': food['name'] ?? 'Unknown',
            'nutrition': food['nutrition'] ?? {
              'calories': 0,
              'sugar': 0,
              'protein': 0,
              'fat': 0,
              'sodium': 0,
              'carbohydrates': 0,
            },
          }).toList();

          final markersPath = '${foodPhotosDir.path}/food_$timestamp.json';
          final markersFile = File(markersPath);
          await markersFile.writeAsString(jsonEncode(markers));

          setState(() {
            _isLoading = false;
          });

          final nutrition = analysisResult['nutrition'] as Map<String, dynamic>? ?? {};
          final foodNames = foods.map((food) => food['name'] ?? 'Unknown').toList();

          // 영양 데이터 저장
          await NutritionData.saveNutritionData(
            date: DateTime.now(),
            calories: (nutrition['calories'] ?? 0).toDouble(),
            sodium: (nutrition['sodium'] ?? 0).toDouble(),
            sugar: (nutrition['sugar'] ?? 0).toDouble(),
            carbohydrates: (nutrition['carbohydrates'] ?? 0).toDouble(),
            imagePath: savedPath,
          );

          showDialog(
            context: context,
            barrierDismissible: false,
            builder: (context) => AlertDialog(
              title: const Text(
                '음식 분석 완료',
                style: TextStyle(
                  fontSize: 20,
                  fontWeight: FontWeight.bold,
                ),
              ),
              content: SingleChildScrollView(
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      '영양 정보',
                      style: TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.bold,
                        color: Colors.blue,
                      ),
                    ),
                    const SizedBox(height: 12),
                    _buildNutritionRow('칼로리', '${nutrition['calories'] ?? 0}', 'kcal'),
                    const SizedBox(height: 8),
                    _buildNutritionRow('당', '${nutrition['sugar'] ?? 0}', 'g'),
                    const SizedBox(height: 8),
                    _buildNutritionRow('단백질', '${nutrition['protein'] ?? 0}', 'g'),
                    const SizedBox(height: 8),
                    _buildNutritionRow('지방', '${nutrition['fat'] ?? 0}', 'g'),
                    const SizedBox(height: 8),
                    _buildNutritionRow('나트륨', '${nutrition['sodium'] ?? 0}', 'mg'),
                    const SizedBox(height: 20),
                    const Divider(),
                    const SizedBox(height: 12),
                    const Text(
                      '인식된 음식',
                      style: TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.bold,
                        color: Colors.orange,
                      ),
                    ),
                    const SizedBox(height: 8),
                    if (foodNames.isEmpty)
                      const Text(
                        '감지된 음식이 없습니다.',
                        style: TextStyle(
                          fontSize: 14,
                          color: Colors.grey,
                        ),
                      )
                    else
                      ...foodNames.asMap().entries.map((entry) => Padding(
                        padding: const EdgeInsets.symmetric(vertical: 4),
                        child: Row(
                          children: [
                            Container(
                              width: 24,
                              height: 24,
                              decoration: BoxDecoration(
                                color: Colors.orange.shade100,
                                shape: BoxShape.circle,
                              ),
                              child: Center(
                                child: Text(
                                  '${entry.key + 1}',
                                  style: TextStyle(
                                    fontSize: 12,
                                    fontWeight: FontWeight.bold,
                                    color: Colors.orange.shade900,
                                  ),
                                ),
                              ),
                            ),
                            const SizedBox(width: 8),
                            Expanded(
                              child: Text(
                                entry.value,
                                style: const TextStyle(
                                  fontSize: 14,
                                  fontWeight: FontWeight.w500,
                                ),
                              ),
                            ),
                          ],
                        ),
                      )).toList(),
                  ],
                ),
              ),
              actions: [
                TextButton(
                  onPressed: () {
                    Navigator.pop(context);
                  },
                  child: const Text('닫기'),
                ),
                TextButton(
                  onPressed: () {
                    Navigator.pop(context);
                    showDialog(
                      context: context,
                      builder: (context) => PhotoPreviewDialog(imagePath: savedPath),
                    );
                  },
                  child: const Text('사진 보기'),
                ),
              ],
            ),
          );
        } catch (e) {
          debugPrint('AI 분석 실패: $e');

          final initialMarkers = [
            {'x': 0.3, 'y': 0.4, 'label': 'Food Item 1'},
            {'x': 0.6, 'y': 0.5, 'label': 'Food Item 2'},
            {'x': 0.5, 'y': 0.3, 'label': 'Food Item 3'},
          ];
          final markersPath = '${foodPhotosDir.path}/food_$timestamp.json';
          final markersFile = File(markersPath);
          await markersFile.writeAsString(jsonEncode(initialMarkers));

          setState(() {
            _isLoading = false;
          });

          showDialog(
            context: context,
            builder: (context) => AlertDialog(
              title: const Text('AI 분석 실패'),
              content: Text('음식 분석에 실패했습니다.\n\n오류: $e'),
              actions: [
                TextButton(
                  onPressed: () {
                    Navigator.pop(context);
                    showDialog(
                      context: context,
                      builder: (context) => PhotoPreviewDialog(imagePath: savedPath),
                    );
                  },
                  child: const Text('사진 보기'),
                ),
              ],
            ),
          );
        }
      }
    } catch (e) {
      debugPrint('Error taking picture: $e');
      setState(() {
        _isLoading = false;
      });
    }
  }

  Widget _buildNutritionRow(String label, String value, String unit) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Text(
          label,
          style: const TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.w500,
          ),
        ),
        Text(
          '$value $unit',
          style: const TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.bold,
            color: Colors.green,
          ),
        ),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Camera'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
      ),
      body: _isInitialized
          ? LayoutBuilder(
              builder: (context, constraints) {
                return Listener(
                  behavior: HitTestBehavior.translucent,
                  onPointerDown: (event) {
                    _pointers[event.pointer] = event.position;

                    if (_pointers.length == 2) {
                      final pointerList = _pointers.values.toList();
                      _initialDistance = (pointerList[0] - pointerList[1]).distance;
                      _baseZoom = _currentZoom;
                      setState(() {
                        _showZoomSlider = true;
                      });
                      _zoomSliderTimer?.cancel();
                    }
                  },
                  onPointerMove: (event) {
                    if (_pointers.containsKey(event.pointer)) {
                      _pointers[event.pointer] = event.position;

                      if (_pointers.length == 2 && _initialDistance != null && _initialDistance! > 0) {
                        final pointerList = _pointers.values.toList();
                        final currentDistance = (pointerList[0] - pointerList[1]).distance;
                        final scale = currentDistance / _initialDistance!;

                        final scaleDelta = scale - 1.0;
                        final sensitivityMultiplier = 2.5;
                        final adjustedScale = 1.0 + (scaleDelta * sensitivityMultiplier);

                        final newZoom = (_baseZoom * adjustedScale).clamp(_minZoom, _maxZoom);

                        setState(() {
                          _currentZoom = newZoom;
                        });
                        _cameraController?.setZoomLevel(newZoom);
                      }
                    }
                  },
                  onPointerUp: (event) {
                    _pointers.remove(event.pointer);

                    if (_pointers.length < 2) {
                      _initialDistance = null;

                      if (_showZoomSlider) {
                        _zoomSliderTimer?.cancel();
                        _zoomSliderTimer = Timer(const Duration(milliseconds: 1500), () {
                          if (mounted) {
                            setState(() {
                              _showZoomSlider = false;
                            });
                          }
                        });
                      }
                    }
                  },
                  onPointerCancel: (event) {
                    _pointers.remove(event.pointer);

                    if (_pointers.length < 2) {
                      _initialDistance = null;
                    }
                  },
                  child: Stack(
                    children: [
                      Positioned.fill(
                        child: CameraPreview(_cameraController!),
                      ),

                      Positioned.fill(
                        child: CustomPaint(
                          painter: GridPainter(),
                        ),
                      ),

                      Builder(
                        builder: (context) {
                          if (_homePosition == Offset.zero) {
                            WidgetsBinding.instance.addPostFrameCallback((_) {
                              final screenWidth = MediaQuery.of(context).size.width;
                              final stackHeight = constraints.maxHeight;
                              setState(() {
                                _homePosition = Offset(
                                  screenWidth / 2 - 35,
                                  stackHeight - 40 - 70,
                                );
                              });
                            });
                          }

                          return TweenAnimationBuilder<Offset>(
                            tween: _isResetting
                                ? Tween<Offset>(
                                    begin: _resetStartPosition,
                                    end: _homePosition,
                                  )
                                : Tween<Offset>(
                                    begin: _buttonPosition,
                                    end: _buttonPosition,
                                  ),
                            duration: _isResetting
                                ? const Duration(milliseconds: 500)
                                : Duration.zero,
                            curve: Curves.easeInOutCubic,
                            builder: (context, animatedOffset, child) {
                              final currentOffset = _isResetting ? animatedOffset : _buttonPosition;

                              return Positioned(
                                bottom: !_isPositioned && !_isResetting ? 40 : null,
                                left: !_isPositioned && !_isResetting
                                    ? MediaQuery.of(context).size.width / 2 - 35
                                    : currentOffset.dx - (_isLongPressing ? 5 : 0),
                                top: _isPositioned || _isResetting
                                    ? currentOffset.dy - (_isLongPressing ? 5 : 0)
                                    : null,
                                child: child!,
                              );
                            },
                            child: GestureDetector(
                              onPanDown: (details) {
                                setState(() {
                                  _isLongPressing = true;
                                  _hasMoved = false;
                                  _resetTimerExecuted = false;
                                  _isResetting = false;
                                  _panStartPosition = details.globalPosition;
                                });

                                _resetTimer?.cancel();
                                _resetTimer = Timer(const Duration(milliseconds: 1200), () {
                                  if (mounted && _isLongPressing && !_hasMoved) {
                                    setState(() {
                                      _resetStartPosition = _buttonPosition;
                                      _isResetting = true;
                                      _resetTimerExecuted = true;
                                    });

                                    Future.delayed(const Duration(milliseconds: 500), () {
                                      if (mounted) {
                                        setState(() {
                                          _isPositioned = false;
                                          _buttonPosition = Offset.zero;
                                          _isLongPressing = false;
                                          _isResetting = false;
                                        });
                                      }
                                    });
                                  }
                                });
                              },
                              onPanStart: (details) {
                                // Do nothing here
                              },
                              onPanUpdate: (details) {
                                if (_panStartPosition != null) {
                                  final distance = (details.globalPosition - _panStartPosition!).distance;
                                  if (distance > 15) {
                                    setState(() {
                                      _hasMoved = true;
                                      _isResetting = false;
                                    });
                                    _resetTimer?.cancel();

                                    if (!_isPositioned) {
                                      final currentTop = constraints.maxHeight - 40 - 70;
                                      _buttonPosition = Offset(
                                        MediaQuery.of(context).size.width / 2 - 35,
                                        currentTop,
                                      );
                                      _isPositioned = true;
                                    }
                                  }
                                }

                                if (_hasMoved) {
                                  setState(() {
                                    _buttonPosition = Offset(
                                      (_buttonPosition.dx + details.delta.dx).clamp(
                                        0.0,
                                        MediaQuery.of(context).size.width - 70,
                                      ),
                                      (_buttonPosition.dy + details.delta.dy).clamp(
                                        0.0,
                                        constraints.maxHeight - 70,
                                      ),
                                    );
                                  });
                                }
                              },
                              onPanEnd: (details) {
                                _resetTimer?.cancel();

                                if (!_hasMoved && !_resetTimerExecuted) {
                                  _takePicture();
                                }

                                setState(() {
                                  _isLongPressing = false;
                                  _hasMoved = false;
                                  _resetTimerExecuted = false;
                                  _panStartPosition = null;
                                });
                              },
                              onPanCancel: () {
                                _resetTimer?.cancel();

                                setState(() {
                                  _isLongPressing = false;
                                  _hasMoved = false;
                                  _resetTimerExecuted = false;
                                  _panStartPosition = null;
                                });
                              },
                              child: _isLoading
                                  ? const CircularProgressIndicator(
                                      valueColor: AlwaysStoppedAnimation<Color>(Colors.white),
                                    )
                                  : AnimatedContainer(
                                      duration: const Duration(milliseconds: 200),
                                      width: _isLongPressing ? 80 : 70,
                                      height: _isLongPressing ? 80 : 70,
                                      decoration: BoxDecoration(
                                        shape: BoxShape.circle,
                                        border: Border.all(
                                          color: Colors.white,
                                          width: 4,
                                        ),
                                      ),
                                      child: Container(
                                        margin: const EdgeInsets.all(4),
                                        decoration: const BoxDecoration(
                                          color: Colors.white,
                                          shape: BoxShape.circle,
                                        ),
                                      ),
                                    ),
                            ),
                          );
                        },
                      ),

                      Positioned(
                        bottom: 40,
                        left: MediaQuery.of(context).size.width * 0.75 - 25,
                        child: GestureDetector(
                          onTap: () {
                            showDialog(
                              context: context,
                              builder: (context) => const GalleryDialog(),
                            );
                          },
                          child: Container(
                            width: 50,
                            height: 50,
                            decoration: BoxDecoration(
                              color: Colors.white.withOpacity(0.3),
                              shape: BoxShape.circle,
                            ),
                            child: const Icon(
                              Icons.folder,
                              color: Colors.white,
                              size: 30,
                            ),
                          ),
                        ),
                      ),

                      if (_showZoomSlider)
                        Positioned(
                          bottom: 120,
                          left: MediaQuery.of(context).size.width * 0.2,
                          right: MediaQuery.of(context).size.width * 0.2,
                          child: Container(
                            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                            decoration: BoxDecoration(
                              color: Colors.black.withOpacity(0.2),
                              borderRadius: BorderRadius.circular(20),
                            ),
                            child: Column(
                              mainAxisSize: MainAxisSize.min,
                              children: [
                                Text(
                                  '${_currentZoom.toStringAsFixed(1)}x',
                                  style: const TextStyle(
                                    color: Colors.white,
                                    fontSize: 12,
                                    fontWeight: FontWeight.bold,
                                  ),
                                ),
                                const SizedBox(height: 4),
                                Row(
                                  children: [
                                    Text(
                                      '${_minZoom.toStringAsFixed(1)}x',
                                      style: const TextStyle(
                                        color: Colors.white,
                                        fontSize: 10,
                                        fontWeight: FontWeight.bold,
                                      ),
                                    ),
                                    const SizedBox(width: 6),
                                    Expanded(
                                      child: SliderTheme(
                                        data: SliderThemeData(
                                          activeTrackColor: Colors.white,
                                          inactiveTrackColor: Colors.white.withOpacity(0.3),
                                          thumbColor: Colors.white,
                                          thumbShape: const RoundSliderThumbShape(
                                            enabledThumbRadius: 6,
                                          ),
                                          overlayShape: const RoundSliderOverlayShape(
                                            overlayRadius: 12,
                                          ),
                                          trackHeight: 2,
                                        ),
                                        child: Slider(
                                          value: _currentZoom,
                                          min: _minZoom,
                                          max: _maxZoom,
                                          onChanged: (value) {
                                            setState(() {
                                              _currentZoom = value;
                                            });

                                            _cameraController?.setZoomLevel(value);

                                            _zoomSliderTimer?.cancel();
                                            _zoomSliderTimer = Timer(const Duration(milliseconds: 1500), () {
                                              if (mounted) {
                                                setState(() {
                                                  _showZoomSlider = false;
                                                });
                                              }
                                            });
                                          },
                                        ),
                                      ),
                                    ),
                                    const SizedBox(width: 6),
                                    Text(
                                      '${_maxZoom.toStringAsFixed(1)}x',
                                      style: const TextStyle(
                                        color: Colors.white,
                                        fontSize: 10,
                                        fontWeight: FontWeight.bold,
                                      ),
                                    ),
                                  ],
                                ),
                              ],
                            ),
                          ),
                        ),
                    ],
                  ),
                );
              },
            )
          : const Center(
              child: CircularProgressIndicator(),
            ),
    );
  }
}

// Supporting classes remain the same...
class GalleryDialog extends StatefulWidget {
  const GalleryDialog({super.key});

  @override
  State<GalleryDialog> createState() => _GalleryDialogState();
}

class _GalleryDialogState extends State<GalleryDialog> {
  List<String> _imagePaths = [];
  bool _isLoading = true;
  bool _isSelectionMode = false;
  Set<String> _selectedImages = {};
  final Map<String, GlobalKey> _itemKeys = {};
  final Map<String, List<Map<String, dynamic>>> _imageMarkers = {};
  bool _isDragging = false;
  ClaudeService? _claudeService;

  @override
  void initState() {
    super.initState();
    final apiKey = dotenv.env['CLAUDE_API_KEY'];
    _claudeService = ClaudeService(apiKey ?? '');
    _loadImages();
  }

  Future<void> _loadImages() async {
    try {
      final directory = await getApplicationDocumentsDirectory();
      final foodPhotosDir = Directory('${directory.path}/temp_food_photos');

      if (await foodPhotosDir.exists()) {
        final files = foodPhotosDir
            .listSync()
            .where((item) => item is File && item.path.endsWith('.jpg'))
            .map((item) => item.path)
            .toList();

        files.sort((a, b) => b.compareTo(a));

        final Map<String, List<Map<String, dynamic>>> markersMap = {};
        for (final imagePath in files) {
          try {
            final markersPath = imagePath.replaceAll('.jpg', '.json');
            final markersFile = File(markersPath);
            if (await markersFile.exists()) {
              final jsonString = await markersFile.readAsString();
              final List<dynamic> decoded = jsonDecode(jsonString);
              markersMap[imagePath] = decoded.map((e) => Map<String, dynamic>.from(e)).toList();
            } else {
              markersMap[imagePath] = [];
            }
          } catch (e) {
            debugPrint('Error loading markers for $imagePath: $e');
            markersMap[imagePath] = [];
          }
        }

        if (mounted) {
          setState(() {
            _imagePaths = files;
            _imageMarkers.clear();
            _imageMarkers.addAll(markersMap);
            _isLoading = false;
            for (final path in files) {
              if (!_itemKeys.containsKey(path)) {
                _itemKeys[path] = GlobalKey();
              }
            }
          });
        }
      } else {
        if (mounted) {
          setState(() {
            _isLoading = false;
          });
        }
      }
    } catch (e) {
      debugPrint('Error loading images: $e');
      if (mounted) {
        setState(() {
          _isLoading = false;
        });
      }
    }
  }

  void _toggleSelection(String imagePath) {
    setState(() {
      if (_selectedImages.contains(imagePath)) {
        _selectedImages.remove(imagePath);
        if (_selectedImages.isEmpty) {
          _isSelectionMode = false;
        }
      } else {
        _selectedImages.add(imagePath);
      }
    });
  }

  Future<void> _deleteSelectedImages() async {
    try {
      for (String imagePath in _selectedImages) {
        // Delete image file
        final imageFile = File(imagePath);
        if (await imageFile.exists()) {
          await imageFile.delete();
        }

        // Delete corresponding JSON file
        final jsonPath = imagePath.replaceAll('.jpg', '.json');
        final jsonFile = File(jsonPath);
        if (await jsonFile.exists()) {
          await jsonFile.delete();
        }
      }

      setState(() {
        _imagePaths.removeWhere((path) => _selectedImages.contains(path));
        _selectedImages.clear();
        _isSelectionMode = false;
      });

      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('선택된 사진이 삭제되었습니다')),
        );
      }
    } catch (e) {
      debugPrint('Error deleting images: $e');
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('사진 삭제 중 오류가 발생했습니다')),
        );
      }
    }
  }

  Future<void> _addImageFromGallery() async {
    try {
      final ImagePicker picker = ImagePicker();
      final XFile? image = await picker.pickImage(source: ImageSource.gallery);

      if (image != null) {
        // 로딩 상태 표시
        setState(() {
          _isLoading = true;
        });

        final directory = await getApplicationDocumentsDirectory();
        final foodPhotosDir = Directory('${directory.path}/temp_food_photos');
        if (!await foodPhotosDir.exists()) {
          await foodPhotosDir.create(recursive: true);
        }

        final timestamp = DateTime.now().millisecondsSinceEpoch;
        final savedPath = '${foodPhotosDir.path}/food_$timestamp.jpg';
        await File(image.path).copy(savedPath);

        // AI로 음식 분석
        try {
          if (_claudeService == null) {
            throw Exception('Claude 서비스가 초기화되지 않았습니다');
          }
          final analysisResult = await _claudeService!.analyzeFoodImage(File(savedPath));

          // 분석된 음식들로 마커 생성
          final foods = analysisResult['foods'] as List<dynamic>? ?? [];
          final markers = foods.map((food) => {
            'x': food['x'] ?? 0.5,
            'y': food['y'] ?? 0.5,
            'label': food['name'] ?? 'Unknown',
            'nutrition': food['nutrition'] ?? {
              'calories': 0,
              'sugar': 0,
              'protein': 0,
              'fat': 0,
              'sodium': 0,
              'carbohydrates': 0,
            },
          }).toList();

          // 마커 저장
          final markersPath = '${foodPhotosDir.path}/food_$timestamp.json';
          final markersFile = File(markersPath);
          await markersFile.writeAsString(jsonEncode(markers));

          // 영양 데이터 저장
          final nutrition = analysisResult['nutrition'] as Map<String, dynamic>? ?? {};
          await NutritionData.saveNutritionData(
            date: DateTime.now(),
            calories: (nutrition['calories'] ?? 0).toDouble(),
            sodium: (nutrition['sodium'] ?? 0).toDouble(),
            sugar: (nutrition['sugar'] ?? 0).toDouble(),
            carbohydrates: (nutrition['carbohydrates'] ?? 0).toDouble(),
            imagePath: savedPath,
          );

          setState(() {
            _isLoading = false;
          });

          await _loadImages();

          if (mounted) {
            // 분석 결과 팝업 표시
            final foodNames = foods.map((food) => food['name'] ?? 'Unknown').toList();

            showDialog(
              context: context,
              barrierDismissible: false,
              builder: (context) => AlertDialog(
                title: const Text(
                  '음식 분석 완료',
                  style: TextStyle(
                    fontSize: 20,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                content: SingleChildScrollView(
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text(
                        '영양 정보',
                        style: TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.bold,
                          color: Colors.blue,
                        ),
                      ),
                      const SizedBox(height: 12),
                      _buildNutritionRow('칼로리', '${nutrition['calories'] ?? 0}', 'kcal'),
                      const SizedBox(height: 8),
                      _buildNutritionRow('당', '${nutrition['sugar'] ?? 0}', 'g'),
                      const SizedBox(height: 8),
                      _buildNutritionRow('단백질', '${nutrition['protein'] ?? 0}', 'g'),
                      const SizedBox(height: 8),
                      _buildNutritionRow('지방', '${nutrition['fat'] ?? 0}', 'g'),
                      const SizedBox(height: 8),
                      _buildNutritionRow('나트륨', '${nutrition['sodium'] ?? 0}', 'mg'),
                      const SizedBox(height: 20),
                      const Divider(),
                      const SizedBox(height: 12),
                      const Text(
                        '인식된 음식',
                        style: TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.bold,
                          color: Colors.orange,
                        ),
                      ),
                      const SizedBox(height: 8),
                      if (foodNames.isEmpty)
                        const Text(
                          '감지된 음식이 없습니다.',
                          style: TextStyle(
                            fontSize: 14,
                            color: Colors.grey,
                          ),
                        )
                      else
                        ...foodNames.asMap().entries.map((entry) => Padding(
                          padding: const EdgeInsets.symmetric(vertical: 4),
                          child: Row(
                            children: [
                              Container(
                                width: 24,
                                height: 24,
                                decoration: BoxDecoration(
                                  color: Colors.orange.shade100,
                                  shape: BoxShape.circle,
                                ),
                                child: Center(
                                  child: Text(
                                    '${entry.key + 1}',
                                    style: TextStyle(
                                      fontSize: 12,
                                      fontWeight: FontWeight.bold,
                                      color: Colors.orange.shade900,
                                    ),
                                  ),
                                ),
                              ),
                              const SizedBox(width: 8),
                              Expanded(
                                child: Text(
                                  entry.value,
                                  style: const TextStyle(
                                    fontSize: 14,
                                    fontWeight: FontWeight.w500,
                                  ),
                                ),
                              ),
                            ],
                          ),
                        )).toList(),
                    ],
                  ),
                ),
                actions: [
                  TextButton(
                    onPressed: () {
                      Navigator.pop(context);
                    },
                    child: const Text('확인'),
                  ),
                ],
              ),
            );
          }
        } catch (e) {
          debugPrint('AI 분석 실패: $e');

          // AI 분석 실패 시 기본 마커 사용
          final initialMarkers = [
            {'x': 0.3, 'y': 0.4, 'label': 'Food Item 1'},
            {'x': 0.6, 'y': 0.5, 'label': 'Food Item 2'},
            {'x': 0.5, 'y': 0.3, 'label': 'Food Item 3'},
          ];
          final markersPath = '${foodPhotosDir.path}/food_$timestamp.json';
          final markersFile = File(markersPath);
          await markersFile.writeAsString(jsonEncode(initialMarkers));

          setState(() {
            _isLoading = false;
          });

          await _loadImages();

          if (mounted) {
            showDialog(
              context: context,
              builder: (context) => AlertDialog(
                title: const Text('AI 분석 실패'),
                content: Text('음식 분석에 실패했습니다.\n\n오류: $e'),
                actions: [
                  TextButton(
                    onPressed: () => Navigator.pop(context),
                    child: const Text('확인'),
                  ),
                ],
              ),
            );
          }
        }
      }
    } catch (e) {
      debugPrint('Error adding image: $e');
      setState(() {
        _isLoading = false;
      });
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('사진 추가 중 오류가 발생했습니다')),
        );
      }
    }
  }

  Widget _buildNutritionRow(String label, String value, String unit) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Text(
          label,
          style: const TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.w500,
          ),
        ),
        Text(
          '$value $unit',
          style: const TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.bold,
            color: Colors.green,
          ),
        ),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    return Dialog(
      backgroundColor: Colors.transparent,
      insetPadding: const EdgeInsets.all(20),
      child: Container(
        width: MediaQuery.of(context).size.width * 0.9,
        height: MediaQuery.of(context).size.height * 0.8,
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(12),
        ),
        child: Column(
          children: [
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
              decoration: BoxDecoration(
                color: Theme.of(context).colorScheme.inversePrimary,
                borderRadius: const BorderRadius.only(
                  topLeft: Radius.circular(12),
                  topRight: Radius.circular(12),
                ),
              ),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  const Text(
                    'Food Photos',
                    style: TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  Row(
                    children: [
                      if (_isSelectionMode)
                        IconButton(
                          icon: const Icon(Icons.delete),
                          onPressed: () async {
                            final confirm = await showDialog<bool>(
                              context: context,
                              builder: (context) => AlertDialog(
                                title: const Text('사진 삭제'),
                                content: Text('선택된 ${_selectedImages.length}개의 사진을 삭제하시겠습니까?'),
                                actions: [
                                  TextButton(
                                    onPressed: () => Navigator.pop(context, false),
                                    child: const Text('취소'),
                                  ),
                                  TextButton(
                                    onPressed: () => Navigator.pop(context, true),
                                    child: const Text('삭제'),
                                  ),
                                ],
                              ),
                            );
                            if (confirm == true) {
                              await _deleteSelectedImages();
                            }
                          },
                        ),
                      IconButton(
                        icon: const Icon(Icons.add),
                        onPressed: _addImageFromGallery,
                      ),
                      IconButton(
                        icon: const Icon(Icons.close),
                        onPressed: () => Navigator.pop(context),
                      ),
                    ],
                  ),
                ],
              ),
            ),
            Expanded(
              child: _isLoading
                  ? const Center(child: CircularProgressIndicator())
                  : _imagePaths.isEmpty
                      ? const Center(
                          child: Column(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              Icon(Icons.photo_library, size: 80, color: Colors.grey),
                              SizedBox(height: 16),
                              Text(
                                '저장된 사진이 없습니다',
                                style: TextStyle(fontSize: 18, color: Colors.grey),
                              ),
                            ],
                          ),
                        )
                      : GridView.builder(
                          padding: const EdgeInsets.all(8),
                          gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                            crossAxisCount: 3,
                            crossAxisSpacing: 8,
                            mainAxisSpacing: 8,
                          ),
                          itemCount: _imagePaths.length,
                          itemBuilder: (context, index) {
                            final imagePath = _imagePaths[index];
                            final isSelected = _selectedImages.contains(imagePath);
                            final itemKey = _itemKeys[imagePath]!;

                            return AnimatedScale(
                              key: itemKey,
                              scale: isSelected ? 0.9 : 1.0,
                              duration: const Duration(milliseconds: 150),
                              curve: Curves.easeOut,
                              child: GestureDetector(
                                behavior: HitTestBehavior.opaque,
                                onTap: () {
                                  if (_isSelectionMode) {
                                    _toggleSelection(imagePath);
                                  } else {
                                    showDialog(
                                      context: context,
                                      builder: (context) => PhotoPreviewDialog(
                                        imagePath: imagePath,
                                      ),
                                    );
                                  }
                                },
                                onLongPressStart: (details) {
                                  setState(() {
                                    _isSelectionMode = true;
                                    _isDragging = true;
                                    if (!_selectedImages.contains(imagePath)) {
                                      _selectedImages.add(imagePath);
                                    }
                                  });
                                },
                                onLongPressMoveUpdate: (details) {
                                  if (_isDragging) {
                                    // Check all items to see which one is under the pointer
                                    for (final path in _imagePaths) {
                                      final key = _itemKeys[path];
                                      if (key?.currentContext != null) {
                                        final RenderBox? box = key!.currentContext!.findRenderObject() as RenderBox?;
                                        if (box != null) {
                                          final position = box.localToGlobal(Offset.zero);
                                          final size = box.size;

                                          // Check if pointer is within this item
                                          if (details.globalPosition.dx >= position.dx &&
                                              details.globalPosition.dx <= position.dx + size.width &&
                                              details.globalPosition.dy >= position.dy &&
                                              details.globalPosition.dy <= position.dy + size.height) {
                                            if (!_selectedImages.contains(path)) {
                                              setState(() {
                                                _selectedImages.add(path);
                                              });
                                            }
                                            break; // Found the item, no need to check others
                                          }
                                        }
                                      }
                                    }
                                  }
                                },
                                onLongPressEnd: (details) {
                                  setState(() {
                                    _isDragging = false;
                                  });
                                },
                                child: Container(
                                  color: Colors.transparent,
                                  child: LayoutBuilder(
                                    builder: (context, constraints) {
                                      final markers = _imageMarkers[imagePath] ?? [];
                                      return Stack(
                                        children: [
                                          ClipRRect(
                                            borderRadius: BorderRadius.circular(8),
                                            child: kIsWeb
                                                ? Image.network(
                                                    imagePath,
                                                    fit: BoxFit.cover,
                                                    width: double.infinity,
                                                    height: double.infinity,
                                                    errorBuilder: (context, error, stackTrace) {
                                                      return const Icon(Icons.broken_image);
                                                    },
                                                  )
                                                : Image.file(
                                                    File(imagePath),
                                                    fit: BoxFit.cover,
                                                    width: double.infinity,
                                                    height: double.infinity,
                                                  ),
                                          ),
                                          // Display markers as small dots
                                          ...markers.map((marker) {
                                            final x = (marker['x'] as num).toDouble();
                                            final y = (marker['y'] as num).toDouble();
                                            return Positioned(
                                              left: constraints.maxWidth * x - 6,
                                              top: constraints.maxHeight * y - 6,
                                              child: IgnorePointer(
                                                child: Container(
                                                  width: 12,
                                                  height: 12,
                                                  decoration: BoxDecoration(
                                                    color: Colors.red.withOpacity(0.8),
                                                    shape: BoxShape.circle,
                                                    border: Border.all(
                                                      color: Colors.white,
                                                      width: 1.5,
                                                    ),
                                                    boxShadow: [
                                                      BoxShadow(
                                                        color: Colors.black.withOpacity(0.3),
                                                        blurRadius: 2,
                                                        spreadRadius: 0.5,
                                                      ),
                                                    ],
                                                  ),
                                                ),
                                              ),
                                            );
                                          }).toList(),
                                          if (isSelected)
                                            Positioned.fill(
                                              child: IgnorePointer(
                                                child: Container(
                                                  decoration: BoxDecoration(
                                                    color: Colors.blue.withOpacity(0.5),
                                                    borderRadius: BorderRadius.circular(8),
                                                    border: Border.all(
                                                      color: Colors.blue,
                                                      width: 3,
                                                    ),
                                                  ),
                                                  child: const Icon(
                                                    Icons.check_circle,
                                                    color: Colors.white,
                                                    size: 40,
                                                  ),
                                                ),
                                              ),
                                            ),
                                        ],
                                      );
                                    },
                                  ),
                                ),
                              ),
                            );
                          },
                        ),
            ),

            // Selection mode info
            if (_isSelectionMode)
              Container(
                padding: const EdgeInsets.all(12),
                color: Colors.grey[200],
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Text(
                      '${_selectedImages.length}개 선택됨',
                      style: const TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    TextButton(
                      onPressed: () {
                        setState(() {
                          _selectedImages.clear();
                          _isSelectionMode = false;
                        });
                      },
                      child: const Text('취소'),
                    ),
                  ],
                ),
              ),
          ],
        ),
      ),
    );
  }
}

class PhotoPreviewDialog extends StatefulWidget {
  final String imagePath;

  const PhotoPreviewDialog({super.key, required this.imagePath});

  @override
  State<PhotoPreviewDialog> createState() => _PhotoPreviewDialogState();
}

class _PhotoPreviewDialogState extends State<PhotoPreviewDialog> {
  List<Map<String, dynamic>> _markers = [];
  bool _isLoading = true;

  @override
  void initState() {
    super.initState();
    _loadMarkers();
  }

  Future<void> _loadMarkers() async {
    try {
      final markersPath = widget.imagePath.replaceAll('.jpg', '.json');
      final markersFile = File(markersPath);

      if (await markersFile.exists()) {
        final jsonString = await markersFile.readAsString();
        final List<dynamic> decoded = jsonDecode(jsonString);
        setState(() {
          _markers = decoded.map((e) => Map<String, dynamic>.from(e)).toList();
          _isLoading = false;
        });
      } else {
        setState(() {
          _isLoading = false;
        });
      }
    } catch (e) {
      debugPrint('Error loading markers: $e');
      setState(() {
        _isLoading = false;
      });
    }
  }

  Widget _buildNutritionRow(String label, String value, String unit) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Text(
          label,
          style: const TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.w500,
          ),
        ),
        Text(
          '$value $unit',
          style: const TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.bold,
            color: Colors.green,
          ),
        ),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    if (_isLoading) {
      return const Dialog(
        backgroundColor: Colors.transparent,
        child: Center(
          child: CircularProgressIndicator(color: Colors.white),
        ),
      );
    }
    return Dialog(
      backgroundColor: Colors.transparent,
      insetPadding: const EdgeInsets.all(20),
      child: SizedBox(
        width: MediaQuery.of(context).size.width * 0.9,
        height: MediaQuery.of(context).size.height * 0.85,
        child: Stack(
          children: [
            Center(
              child: ClipRRect(
                borderRadius: BorderRadius.circular(12),
                child: kIsWeb
                    ? Image.network(
                        widget.imagePath,
                        fit: BoxFit.contain,
                        errorBuilder: (context, error, stackTrace) {
                          return const Icon(Icons.broken_image);
                        },
                      )
                    : Image.file(
                        File(widget.imagePath),
                        fit: BoxFit.contain,
                      ),
              ),
            ),

            ...(_markers.asMap().entries.map((entry) {
              final marker = entry.value;
              return Positioned(
                left: MediaQuery.of(context).size.width * 0.9 * marker['x'],
                top: MediaQuery.of(context).size.height * 0.85 * marker['y'],
                child: GestureDetector(
                  onTap: () {
                    // 마커 클릭 시 해당 음식의 영양 정보 표시
                    final nutrition = marker['nutrition'] as Map<String, dynamic>?;
                    if (nutrition != null) {
                      showDialog(
                        context: context,
                        builder: (context) => AlertDialog(
                          title: Text(
                            marker['label'] ?? 'Unknown',
                            style: const TextStyle(
                              fontSize: 20,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                          content: SingleChildScrollView(
                            child: Column(
                              mainAxisSize: MainAxisSize.min,
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                const Text(
                                  '영양 정보',
                                  style: TextStyle(
                                    fontSize: 16,
                                    fontWeight: FontWeight.bold,
                                    color: Colors.blue,
                                  ),
                                ),
                                const SizedBox(height: 12),
                                _buildNutritionRow('칼로리', '${nutrition['calories'] ?? 0}', 'kcal'),
                                const SizedBox(height: 8),
                                _buildNutritionRow('당', '${nutrition['sugar'] ?? 0}', 'g'),
                                const SizedBox(height: 8),
                                _buildNutritionRow('단백질', '${nutrition['protein'] ?? 0}', 'g'),
                                const SizedBox(height: 8),
                                _buildNutritionRow('지방', '${nutrition['fat'] ?? 0}', 'g'),
                                const SizedBox(height: 8),
                                _buildNutritionRow('나트륨', '${nutrition['sodium'] ?? 0}', 'mg'),
                                const SizedBox(height: 8),
                                _buildNutritionRow('탄수화물', '${nutrition['carbohydrates'] ?? 0}', 'g'),
                              ],
                            ),
                          ),
                          actions: [
                            TextButton(
                              onPressed: () => Navigator.pop(context),
                              child: const Text('확인'),
                            ),
                          ],
                        ),
                      );
                    }
                  },
                  child: Column(
                    children: [
                      Container(
                        width: 40,
                        height: 40,
                        decoration: BoxDecoration(
                          color: Colors.red.withOpacity(0.7),
                          shape: BoxShape.circle,
                          border: Border.all(
                            color: Colors.white,
                            width: 2,
                          ),
                        ),
                        child: const Icon(
                          Icons.restaurant,
                          color: Colors.white,
                          size: 24,
                        ),
                      ),
                      const SizedBox(height: 4),
                      Container(
                        padding: const EdgeInsets.symmetric(
                          horizontal: 8,
                          vertical: 4,
                        ),
                        decoration: BoxDecoration(
                          color: Colors.black87,
                          borderRadius: BorderRadius.circular(4),
                        ),
                        child: Text(
                          marker['label'],
                          style: const TextStyle(
                            color: Colors.white,
                            fontSize: 12,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              );
            }).toList()),

            Positioned(
              top: 30,
              right: 10,
              child: GestureDetector(
                onTap: () => Navigator.pop(context),
                child: Container(
                  width: 40,
                  height: 40,
                  decoration: BoxDecoration(
                    color: Colors.black54,
                    shape: BoxShape.circle,
                  ),
                  child: const Icon(
                    Icons.close,
                    color: Colors.white,
                    size: 24,
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class GridPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.white.withOpacity(0.5)
      ..strokeWidth = 1.0
      ..style = PaintingStyle.stroke;

    final verticalStep = size.width / 3;
    canvas.drawLine(
      Offset(verticalStep, 0),
      Offset(verticalStep, size.height),
      paint,
    );
    canvas.drawLine(
      Offset(verticalStep * 2, 0),
      Offset(verticalStep * 2, size.height),
      paint,
    );

    final horizontalStep = size.height / 3;
    canvas.drawLine(
      Offset(0, horizontalStep),
      Offset(size.width, horizontalStep),
      paint,
    );
    canvas.drawLine(
      Offset(0, horizontalStep * 2),
      Offset(size.width, horizontalStep * 2),
      paint,
    );
  }

  @override
  bool shouldRepaint(CustomPainter oldDelegate) => false;
}