import 'package:flutter/material.dart';
import 'screens/nutri_login_screen.dart';

void main() {
  runApp(const NutriScanApp());
}

class NutriScanApp extends StatelessWidget {
  const NutriScanApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'NutriScan AI',
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF4A5D4E),
          brightness: Brightness.light,
        ),
      ),
      home: const NutriLoginScreen(
        palette: 'sage',
        isDarkMode: false,
      ),
    );
  }
}
