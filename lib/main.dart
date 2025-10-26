import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter/rendering.dart';
import 'package:camera/camera.dart';
import 'dart:io';
import 'dart:convert';
import 'dart:async';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as path;

void main() {
  // Disable debug banners and overlays
  WidgetsFlutterBinding.ensureInitialized();

  // Disable debug paint borders
  debugPaintSizeEnabled = false;
  debugPaintBaselinesEnabled = false;
  debugPaintPointersEnabled = false;
  debugPaintLayerBordersEnabled = false;
  debugRepaintRainbowEnabled = false;

  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      debugShowMaterialGrid: false,
      showPerformanceOverlay: false,
      title: 'Food App',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.blue),
        useMaterial3: true,
        focusColor: Colors.transparent,
        highlightColor: Colors.transparent,
        splashColor: Colors.transparent.withOpacity(0.1),
      ),
      home: const MainScreen(),
    );
  }
}

class MainScreen extends StatefulWidget {
  const MainScreen({super.key});

  @override
  State<MainScreen> createState() => _MainScreenState();
}

class _MainScreenState extends State<MainScreen> {
  int _currentIndex = 0;

  final List<Widget> _pages = [
    const HomePage(),
    const CameraPage(),
    const PlaceholderPage(title: 'Tab 3'),
    const UserInfoPage(),
  ];

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
            icon: Icon(Icons.circle),
            label: 'null',
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
  const HomePage({super.key});

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
              title: const Text('null'),
              onTap: () {
                Navigator.pop(context);
                _showPlaceholderDialog('메뉴 1');
              },
            ),
            ListTile(
              leading: const Icon(Icons.circle_outlined),
              title: const Text('null'),
              onTap: () {
                Navigator.pop(context);
                _showPlaceholderDialog('메뉴 2');
              },
            ),
            ListTile(
              leading: const Icon(Icons.circle_outlined),
              title: const Text('null'),
              onTap: () {
                Navigator.pop(context);
                _showPlaceholderDialog('메뉴 3');
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
                  child: InkWell(
                    onTap: () => _showPlaceholderDialog('$dayNumber일'),
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
                          // Blank space for future information
                          Container(
                            width: 30,
                            height: 20,
                            decoration: BoxDecoration(
                              color: Colors.grey.shade100,
                              borderRadius: BorderRadius.circular(4),
                            ),
                          ),
                        ],
                      ),
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

class PlaceholderPage extends StatelessWidget {
  final String title;

  const PlaceholderPage({super.key, required this.title});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(title),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(
              Icons.construction,
              size: 100,
              color: Colors.grey[400],
            ),
            const SizedBox(height: 20),
            Text(
              title,
              style: const TextStyle(
                fontSize: 24,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 10),
            Text(
              '준비 중입니다',
              style: TextStyle(
                fontSize: 16,
                color: Colors.grey[600],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

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

  // Draggable button state
  Offset _buttonPosition = Offset.zero;
  bool _isPositioned = false; // Whether button has been moved from initial position
  bool _isLongPressing = false;
  Timer? _resetTimer;
  Offset? _panStartPosition;
  bool _hasMoved = false;
  bool _resetTimerExecuted = false;
  bool _isResetting = false; // Flag for reset animation
  Offset _resetStartPosition = Offset.zero; // Start position for reset animation
  Offset _homePosition = Offset.zero; // Home position (center bottom)

  // Zoom state
  double _currentZoom = 1.0;
  double _minZoom = 1.0;
  double _maxZoom = 1.0;
  bool _showZoomSlider = false;
  Timer? _zoomSliderTimer;
  double _baseZoom = 1.0; // Base zoom level for pinch gesture

  // Pinch gesture tracking
  double? _initialDistance;
  Map<int, Offset> _pointers = {}; // Track all active pointers by ID

  @override
  void initState() {
    super.initState();
    _initializeCamera();
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

        // Get zoom capabilities - use device's actual range
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
        // Auto-save the picture
        final directory = await getApplicationDocumentsDirectory();
        final foodPhotosDir = Directory('${directory.path}/temp_food_photos');
        if (!await foodPhotosDir.exists()) {
          await foodPhotosDir.create(recursive: true);
        }

        final timestamp = DateTime.now().millisecondsSinceEpoch;
        final savedPath = '${foodPhotosDir.path}/food_$timestamp.jpg';
        await File(photo.path).copy(savedPath);

        // Save initial markers
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

        // Show preview dialog
        showDialog(
          context: context,
          builder: (context) => PhotoPreviewDialog(imagePath: savedPath),
        );
      }
    } catch (e) {
      debugPrint('Error taking picture: $e');
      setState(() {
        _isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    // Hide system UI overlays
    SystemChrome.setEnabledSystemUIMode(SystemUiMode.manual, overlays: [SystemUiOverlay.bottom]);

    return Scaffold(
      appBar: AppBar(
        title: const Text('Camera'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
      ),
      extendBody: true,
      extendBodyBehindAppBar: false,
      body: _isInitialized
          ? LayoutBuilder(
              builder: (context, constraints) {
                return Listener(
                  behavior: HitTestBehavior.translucent,
                  onPointerDown: (event) {
                    _pointers[event.pointer] = event.position;
                    debugPrint('👆 Pointer down: ${event.pointer}, total pointers: ${_pointers.length}');

                    if (_pointers.length == 2) {
                      // Two fingers detected, start zoom
                      final pointerList = _pointers.values.toList();
                      _initialDistance = (pointerList[0] - pointerList[1]).distance;
                      _baseZoom = _currentZoom;
                      debugPrint('🔍 Two fingers detected, starting zoom. Initial distance: $_initialDistance');
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

                        // Apply sensitivity multiplier
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
                    debugPrint('👆 Pointer up: ${event.pointer}, remaining pointers: ${_pointers.length}');

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
                    debugPrint('👆 Pointer cancel: ${event.pointer}, remaining pointers: ${_pointers.length}');

                    if (_pointers.length < 2) {
                      _initialDistance = null;
                    }
                  },
                  child: Stack(
                    children: [
                      // Camera preview
                      Positioned.fill(
                        child: CameraPreview(_cameraController!),
                      ),

                    // Rule of thirds grid
                    Positioned.fill(
                      child: CustomPaint(
                        painter: GridPainter(),
                      ),
                    ),

                    // Capture button at center bottom (draggable)
                    Builder(
                      builder: (context) {
                        // Calculate home position once
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
                          debugPrint('📸 Camera button onPanDown called');
                          setState(() {
                            _isLongPressing = true;
                            _hasMoved = false;
                            _resetTimerExecuted = false;
                            _isResetting = false; // Not resetting when user touches
                            _panStartPosition = details.globalPosition;
                          });

                          // Start 1.2-second timer when user touches button
                          _resetTimer?.cancel();
                          _resetTimer = Timer(const Duration(milliseconds: 1200), () {
                            if (mounted && _isLongPressing && !_hasMoved) {
                              debugPrint('📸 Reset timer executed - returning to home position');
                              setState(() {
                                _resetStartPosition = _buttonPosition; // Save current position
                                _isResetting = true; // Enable animation for reset
                                _resetTimerExecuted = true;
                              });

                              // After animation duration, finalize the reset
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
                          debugPrint('📸 Camera button onPanStart called');
                          // Do nothing here - position will be set when actually dragging
                        },
                        onPanUpdate: (details) {
                          // Check if user has moved significantly
                          if (_panStartPosition != null) {
                            final distance = (details.globalPosition - _panStartPosition!).distance;
                            if (distance > 15) {
                              // User is dragging (increased threshold to 15 pixels)
                              debugPrint('📸 Camera button dragging - distance: $distance');
                              setState(() {
                                _hasMoved = true;
                                _isResetting = false; // No animation when dragging
                              });
                              _resetTimer?.cancel();

                              // Calculate position on first move
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
                          debugPrint('📸 Camera button onPanEnd - hasMoved: $_hasMoved, resetTimerExecuted: $_resetTimerExecuted');
                          _resetTimer?.cancel();

                          // Only take picture if user didn't move AND timer didn't execute
                          if (!_hasMoved && !_resetTimerExecuted) {
                            debugPrint('📸 Taking picture!');
                            _takePicture();
                          } else {
                            debugPrint('📸 Not taking picture - hasMoved: $_hasMoved, resetTimerExecuted: $_resetTimerExecuted');
                          }

                          setState(() {
                            _isLongPressing = false;
                            _hasMoved = false;
                            _resetTimerExecuted = false;
                            _panStartPosition = null;
                          });
                        },
                        onPanCancel: () {
                          debugPrint('📸 Camera button onPanCancel called');
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

                    // Folder button - between capture button and right edge
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

                    // Zoom slider (horizontal)
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
                              // Current zoom value
                              Text(
                                '${_currentZoom.toStringAsFixed(1)}x',
                                style: const TextStyle(
                                  color: Colors.white,
                                  fontSize: 12,
                                  fontWeight: FontWeight.bold,
                                ),
                              ),
                              const SizedBox(height: 4),

                              // Horizontal slider with labels
                              Row(
                                children: [
                                  // Min zoom label
                                  Text(
                                    '${_minZoom.toStringAsFixed(1)}x',
                                    style: const TextStyle(
                                      color: Colors.white,
                                      fontSize: 10,
                                      fontWeight: FontWeight.bold,
                                    ),
                                  ),
                                  const SizedBox(width: 6),

                                  // Slider
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

                                          // Apply zoom to camera
                                          _cameraController?.setZoomLevel(value);

                                          // Reset auto-hide timer
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
                                  // Max zoom label
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

  @override
  void initState() {
    super.initState();
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

        files.sort((a, b) => b.compareTo(a)); // Sort by newest first

        if (mounted) {
          setState(() {
            _imagePaths = files;
            _isLoading = false;
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
            // Header with close button and delete button
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
                      if (_isSelectionMode && _selectedImages.isNotEmpty)
                        IconButton(
                          icon: const Icon(Icons.delete),
                          onPressed: () {
                            showDialog(
                              context: context,
                              builder: (context) => AlertDialog(
                                title: const Text('삭제 확인'),
                                content: Text('${_selectedImages.length}개의 사진을 삭제하시겠습니까?'),
                                actions: [
                                  TextButton(
                                    onPressed: () => Navigator.pop(context),
                                    child: const Text('취소'),
                                  ),
                                  TextButton(
                                    onPressed: () {
                                      Navigator.pop(context);
                                      _deleteSelectedImages();
                                    },
                                    child: const Text('삭제'),
                                  ),
                                ],
                              ),
                            );
                          },
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

            // Gallery grid
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

                            return GestureDetector(
                              onTap: () {
                                if (_isSelectionMode) {
                                  _toggleSelection(imagePath);
                                } else {
                                  Navigator.pop(context);
                                  showDialog(
                                    context: context,
                                    builder: (context) => PhotoPreviewDialog(
                                      imagePath: imagePath,
                                    ),
                                  );
                                }
                              },
                              onLongPress: () {
                                setState(() {
                                  _isSelectionMode = true;
                                  _toggleSelection(imagePath);
                                });
                              },
                              child: Stack(
                                children: [
                                  ClipRRect(
                                    borderRadius: BorderRadius.circular(8),
                                    child: Image.file(
                                      File(imagePath),
                                      fit: BoxFit.cover,
                                      width: double.infinity,
                                      height: double.infinity,
                                    ),
                                  ),
                                  if (isSelected)
                                    Positioned.fill(
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
                                ],
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

  Future<void> _saveMarkers() async {
    try {
      final markersPath = widget.imagePath.replaceAll('.jpg', '.json');
      final markersFile = File(markersPath);
      await markersFile.writeAsString(jsonEncode(_markers));
    } catch (e) {
      debugPrint('Error saving markers: $e');
    }
  }

  Future<void> _editMarker(int index) async {
    final controller = TextEditingController(text: _markers[index]['label']);

    final result = await showDialog<String>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('음식 이름 수정'),
        content: TextField(
          controller: controller,
          decoration: const InputDecoration(
            hintText: '음식 이름을 입력하세요',
          ),
          autofocus: true,
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('취소'),
          ),
          TextButton(
            onPressed: () => Navigator.pop(context, controller.text),
            child: const Text('확인'),
          ),
        ],
      ),
    );

    if (result != null && result.isNotEmpty) {
      setState(() {
        _markers[index]['label'] = result;
      });
      await _saveMarkers();
    }
  }

  Future<void> _deletePhoto() async {
    final confirm = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('사진 삭제'),
        content: const Text('이 사진을 삭제하시겠습니까?'),
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
      try {
        // Delete image file
        final imageFile = File(widget.imagePath);
        if (await imageFile.exists()) {
          await imageFile.delete();
        }

        // Delete JSON file
        final jsonPath = widget.imagePath.replaceAll('.jpg', '.json');
        final jsonFile = File(jsonPath);
        if (await jsonFile.exists()) {
          await jsonFile.delete();
        }

        if (mounted) {
          Navigator.pop(context); // Close preview dialog
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text('사진이 삭제되었습니다')),
          );
        }
      } catch (e) {
        debugPrint('Error deleting photo: $e');
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text('사진 삭제 중 오류가 발생했습니다')),
          );
        }
      }
    }
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
            // Display the captured image
            Center(
              child: ClipRRect(
                borderRadius: BorderRadius.circular(12),
                child: Image.file(
                  File(widget.imagePath),
                  fit: BoxFit.contain,
                ),
              ),
            ),

            // Test markers overlay
            ...(_markers.asMap().entries.map((entry) {
              final index = entry.key;
              final marker = entry.value;
              return Positioned(
                left: MediaQuery.of(context).size.width * 0.9 * marker['x'],
                top: MediaQuery.of(context).size.height * 0.85 * marker['y'],
                child: GestureDetector(
                  onTap: () => _editMarker(index),
                  child: Column(
                    children: [
                      // Marker icon
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
                      // Label
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

            // Delete button at top left
            Positioned(
              top: 30,
              left: 10,
              child: GestureDetector(
                onTap: _deletePhoto,
                child: Container(
                  width: 40,
                  height: 40,
                  decoration: BoxDecoration(
                    color: Colors.black54,
                    shape: BoxShape.circle,
                  ),
                  child: const Icon(
                    Icons.delete,
                    color: Colors.white,
                    size: 24,
                  ),
                ),
              ),
            ),

            // Close button at top right
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

            // Bottom instructions
            Positioned(
              bottom: 20,
              left: 20,
              right: 20,
              child: Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: Colors.black87,
                  borderRadius: BorderRadius.circular(8),
                ),
                child: const Text(
                  '마커를 탭하여 음식 이름을 추가하거나 수정할 수 있습니다.',
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: 14,
                  ),
                  textAlign: TextAlign.center,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

// Custom painter for camera grid
class GridPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.white.withOpacity(0.5)
      ..strokeWidth = 1.0
      ..style = PaintingStyle.stroke;

    // Draw vertical lines (2 lines dividing into 3 sections)
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

    // Draw horizontal lines (2 lines dividing into 3 sections)
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
