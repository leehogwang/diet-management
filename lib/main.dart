import 'package:flutter/material.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Food App',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepOrange),
        useMaterial3: true,
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
    const PlaceholderPage(title: 'Tab 2'),
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
            icon: Icon(Icons.circle),
            label: 'null',
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
