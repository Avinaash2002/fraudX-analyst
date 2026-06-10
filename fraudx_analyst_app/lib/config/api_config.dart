/// FraudX Analyst - API Configuration
/// =====================================

import 'package:shared_preferences/shared_preferences.dart';

class ApiConfig {
  // ── Choose your environment ────────────────────────────────────────────────
  static const AppEnvironment environment = AppEnvironment.production;
  
  // ── Backend URLs ───────────────────────────────────────────────────────────
  static const String _emulatorUrl = 'http://10.0.2.2:8000';
  static const String _localDeviceUrl = 'http://10.87.151.22:8000';
  static const String _productionUrl = 'https://fraudx-analyst.onrender.com';
  
  // ── Active base URL ────────────────────────────────────────────────────────
  static String get baseUrl {
    switch (environment) {
      case AppEnvironment.emulator:
        return _emulatorUrl;
      case AppEnvironment.localDevice:
        return _localDeviceUrl;
      case AppEnvironment.production:
        return _productionUrl;
    }
  }
  
  // ── API Endpoints ──────────────────────────────────────────────────────────
  static String get predict => '$baseUrl/api/v1/predict';
  static String get models => '$baseUrl/api/v1/models';
  static String get modelsCompare => '$baseUrl/api/v1/models/compare';
  static String get history => '$baseUrl/api/v1/history';
  static String get chat => '$baseUrl/api/v1/chat';
  static String get chatSuggestions => '$baseUrl/api/v1/chat/suggestions';
  static String get trainValidate => '$baseUrl/api/v1/train/validate';
  static String get trainCustom => '$baseUrl/api/v1/train/custom';
  static String get sampleTransaction => '$baseUrl/api/v1/sample-transaction';
  
  // ── Timeout settings ───────────────────────────────────────────────────────
  static const Duration timeout = Duration(seconds: 30);
  
  // ── Device ID (persisted across app restarts) ─────────────────────────────
  static String? _deviceId;
  
  /// Call once at app startup before runApp()
  static Future<void> init() async {
    final prefs = await SharedPreferences.getInstance();
    _deviceId = prefs.getString('device_id');
    if (_deviceId == null) {
      _deviceId = 'FX-${DateTime.now().millisecondsSinceEpoch}';
      await prefs.setString('device_id', _deviceId!);
    }
  }

  static String get deviceId {
    _deviceId ??= 'FX-${DateTime.now().millisecondsSinceEpoch}';
    return _deviceId!;
  }
}

enum AppEnvironment {
  emulator,
  localDevice,
  production,
}
