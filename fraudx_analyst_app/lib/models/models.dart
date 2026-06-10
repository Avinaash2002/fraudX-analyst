/// FraudX Analyst - Data Models
/// ===============================
/// All data structures for API communication

// ══════════════════════════════════════════════════════════════════════════════
//  Prediction Models
// ══════════════════════════════════════════════════════════════════════════════

class PredictRequest {
  final String modelName;
  final double amount;
  final double time;
  final Map<String, double> features; // V1-V28
  final String? deviceId;
  final String? cardNumber;
  final String? location;

  PredictRequest({
    required this.modelName,
    required this.amount,
    required this.time,
    required this.features,
    this.deviceId,
    this.cardNumber,
    this.location,
  });

  Map<String, dynamic> toJson() => {
        'model_name': modelName,
        'amount': amount,
        'time': time,
        ...features,
        if (deviceId != null) 'device_id': deviceId,
        if (cardNumber != null) 'card_number': cardNumber,
        if (location != null) 'location': location,
      };
}

class PredictResponse {
  final String simulationId;
  final String prediction;
  final double riskScore;
  final double confidenceScore;
  final double processingTime;
  final List<TopFeature> topFeatures;
  final List<ShapValue>? shapValues;
  final String aiExplanation;

  PredictResponse({
    required this.simulationId,
    required this.prediction,
    required this.riskScore,
    required this.confidenceScore,
    required this.processingTime,
    required this.topFeatures,
    this.shapValues,
    required this.aiExplanation,
  });

  factory PredictResponse.fromJson(Map<String, dynamic> json) {
    return PredictResponse(
      simulationId: json['simulation_id'],
      prediction: json['prediction'],
      riskScore: (json['risk_score'] as num).toDouble(),
      confidenceScore: (json['confidence_score'] as num).toDouble(),
      processingTime: (json['processing_time'] as num).toDouble(),
      topFeatures: (json['top_features'] as List)
          .map((f) => TopFeature.fromJson(f))
          .toList(),
      shapValues: json['shap_values'] != null
          ? (json['shap_values'] as List)
              .map((s) => ShapValue.fromJson(s))
              .toList()
          : null,
      aiExplanation: json['ai_explanation'],
    );
  }

  bool get isFraud => prediction == 'FRAUD';
  
  String get riskLevel {
    if (riskScore < 0.25) return 'Low';
    if (riskScore < 0.50) return 'Medium';
    if (riskScore < 0.75) return 'High';
    return 'Critical';
  }
}

class TopFeature {
  final String feature;
  final double value;
  final double impact;

  TopFeature({
    required this.feature,
    required this.value,
    required this.impact,
  });

  factory TopFeature.fromJson(Map<String, dynamic> json) {
    return TopFeature(
      feature: json['feature'],
      value: (json['value'] as num).toDouble(),
      impact: (json['impact'] as num).toDouble(),
    );
  }
}

class ShapValue {
  final String feature;
  final double value;
  final double impact;

  ShapValue({
    required this.feature,
    required this.value,
    required this.impact,
  });

  factory ShapValue.fromJson(Map<String, dynamic> json) {
    return ShapValue(
      feature: json['feature'],
      value: (json['value'] as num).toDouble(),
      impact: (json['impact'] as num).toDouble(),
    );
  }
}

// ══════════════════════════════════════════════════════════════════════════════
//  Model Metrics
// ══════════════════════════════════════════════════════════════════════════════

class ModelMetrics {
  final String modelName;
  final double accuracy;
  final double precision;
  final double recall;
  final double f1Score;
  final double aucRoc;
  final String algorithmType;
  final double? trainingTime;
  final String? description;

  ModelMetrics({
    required this.modelName,
    required this.accuracy,
    required this.precision,
    required this.recall,
    required this.f1Score,
    required this.aucRoc,
    required this.algorithmType,
    this.trainingTime,
    this.description,
  });

  factory ModelMetrics.fromJson(Map<String, dynamic> json) {
    return ModelMetrics(
      modelName: json['model_name'],
      accuracy: (json['accuracy'] as num).toDouble(),
      precision: (json['precision'] as num).toDouble(),
      recall: (json['recall'] as num).toDouble(),
      f1Score: (json['f1_score'] as num).toDouble(),
      aucRoc: (json['auc_roc'] as num).toDouble(),
      algorithmType: json['algorithm_type'],
      trainingTime: json['training_time'] != null 
          ? (json['training_time'] as num).toDouble() 
          : null,
      description: json['description'],
    );
  }
}

class ModelsComparisonData {
  final List<String> models;
  final Map<String, List<double>> metrics;

  ModelsComparisonData({
    required this.models,
    required this.metrics,
  });

  factory ModelsComparisonData.fromJson(Map<String, dynamic> json) {
    return ModelsComparisonData(
      models: List<String>.from(json['models']),
      metrics: Map<String, List<double>>.from(
        (json['metrics'] as Map).map(
          (key, value) => MapEntry(
            key,
            (value as List).map((v) => (v as num).toDouble()).toList(),
          ),
        ),
      ),
    );
  }
}

// ══════════════════════════════════════════════════════════════════════════════
//  History
// ══════════════════════════════════════════════════════════════════════════════

class HistoryItem {
  final String simulationId;
  final DateTime timestamp;
  final double transactionAmount;
  final String predictionResult;
  final double riskScore;
  final String? location;
  final String? cardNumber;
  final String? aiExplanation;
  final String? modelUsed;

  HistoryItem({
    required this.simulationId,
    required this.timestamp,
    required this.transactionAmount,
    required this.predictionResult,
    required this.riskScore,
    this.location,
    this.cardNumber,
    this.aiExplanation,
    this.modelUsed,
  });

  factory HistoryItem.fromJson(Map<String, dynamic> json) {
    return HistoryItem(
      simulationId: json['simulation_id'],
      timestamp: DateTime.parse('${json['timestamp']}Z').toLocal(),
      transactionAmount: (json['transaction_amount'] as num).toDouble(),
      predictionResult: json['prediction_result'],
      riskScore: (json['risk_score'] as num).toDouble(),
      location: json['location'],
      cardNumber: json['card_number'],
      aiExplanation: json['ai_explanation'],
      modelUsed: json['model_used'],
    );
  }

  bool get isFraud => predictionResult == 'FRAUD';

  String get prediction => predictionResult;
}

// ══════════════════════════════════════════════════════════════════════════════
//  Chat
// ══════════════════════════════════════════════════════════════════════════════

class ChatMessage {
  final String message;
  final bool isUser;
  final DateTime timestamp;
  final List<String>? sources;

  ChatMessage({
    required this.message,
    required this.isUser,
    required this.timestamp,
    this.sources,
  });
}

class ChatRequest {
  final String message;
  final String? deviceId;
  final String? simulationId;
  final List<Map<String, String>>? chatHistory;

  ChatRequest({
    required this.message,
    this.deviceId,
    this.simulationId,
    this.chatHistory,
  });

  Map<String, dynamic> toJson() => {
        'message': message,
        if (deviceId != null) 'device_id': deviceId,
        if (simulationId != null) 'simulation_id': simulationId,
        if (chatHistory != null) 'chat_history': chatHistory,
      };
}

class ChatResponse {
  final String reply;
  final List<String>? sources;

  ChatResponse({
    required this.reply,
    this.sources,
  });

  factory ChatResponse.fromJson(Map<String, dynamic> json) {
    return ChatResponse(
      reply: json['reply'],
      sources: json['sources'] != null
          ? List<String>.from(json['sources'])
          : null,
    );
  }
}
