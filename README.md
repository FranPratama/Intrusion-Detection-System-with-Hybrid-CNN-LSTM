# Pseudocode Implementasi CNN-LSTM Intrusion Detection System

## 1. Data Loading dan Preprocessing
```
FUNCTION LoadAndPreprocess():
    // Load data dari multiple file CSV
    files ← findAllCSVFiles(directory_path)
    data ← concatenate(readCSV(files))
    
    // Hapus duplikasi dan reset index
    data ← removeDuplicates(data)
    data ← resetIndex(data)
    
    // Konversi kolom kategorikal ke representasi numerik
    FOR EACH column IN data WHERE type(column) is "object":
        column ← convertToCategory(column)
        column ← getCategoryCodes(column)
    END FOR
    
    // Ekstraksi fitur dan label
    X ← data EXCLUDING "label" column
    y ← data["label"]
    
    // Label encoding untuk target
    y ← labelEncode(y)
    
    // Penyeimbangan data dengan upsampling
    majority_class ← subset(data WHERE label = 0)
    minority_class ← subset(data WHERE label = 1)
    minority_upsampled ← resample(minority_class, n_samples=length(majority_class))
    balanced_data ← concatenate(majority_class, minority_upsampled)
    
    // Ekstraksi fitur dan label dari data seimbang
    X ← balanced_data EXCLUDING "label" column
    y ← balanced_data["label"]
    
    // Split training dan testing
    X_train, X_test, y_train, y_test ← splitData(X, y, test_size=0.2)
    
    // Normalisasi data
    scaler ← initializeStandardScaler()
    X_train ← fitTransform(scaler, X_train)
    X_test ← transform(scaler, X_test)
    
    // One-hot encoding untuk label
    num_classes ← countUniqueValues(y)
    y_train ← oneHotEncode(y_train, num_classes)
    y_test ← oneHotEncode(y_test, num_classes)
    
    RETURN X_train, X_test, y_train, y_test, num_classes
END FUNCTION
```

## 2. Konstruksi Model CNN-LSTM
```
FUNCTION BuildModel(input_shape, num_classes):
    // Inisialisasi model sekuensial
    model ← createSequentialModel()
    
    // Menambahkan lapisan konvolusional pertama
    model.add(Conv1D(filters=64, kernel_size=1, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(rate=0.3))
    
    // Menambahkan lapisan konvolusional kedua
    model.add(Conv1D(filters=64, kernel_size=1, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(rate=0.3))
    
    // Menambahkan lapisan konvolusional ketiga
    model.add(Conv1D(filters=64, kernel_size=1, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    
    // Menambahkan lapisan LSTM
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(rate=0.2))
    
    // Menambahkan lapisan fully connected
    model.add(Dense(units=64, activation='relu'))
    model.add(Dropout(rate=0.3))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=num_classes, activation='softmax'))
    
    // Kompilasi model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    RETURN model
END FUNCTION
```

## 3. Pelatihan Model
```
FUNCTION TrainModel(model, X_train, y_train):
    // Konfigurasi callbacks
    checkpoint ← createModelCheckpoint(filepath='best_model', monitor='val_accuracy', save_best_only=TRUE)
    early_stopping ← createEarlyStopping(monitor='val_loss', patience=10)
    
    // Melatih model dengan validasi
    history ← model.fit(
        X_train, 
        y_train, 
        epochs=30, 
        batch_size=32, 
        validation_split=0.2, 
        callbacks=[checkpoint, early_stopping]
    )
    
    // Memuat bobot model terbaik
    model.loadWeights('best_model')
    
    RETURN model, history
END FUNCTION
```

## 4. Evaluasi Model
```
FUNCTION EvaluateModel(model, X_test, y_test):
    // Prediksi data test
    predictions ← model.predict(X_test)
    predictions_binary ← (predictions > 0.5)
    
    // Ekstraksi label untuk evaluasi multi-kelas
    predicted_labels ← argmax(predictions_binary, axis=1)
    true_labels ← argmax(y_test, axis=1)
    
    // Hitung metrik evaluasi
    accuracy ← calculateAccuracy(true_labels, predicted_labels)
    precision ← calculatePrecision(true_labels, predicted_labels, average='weighted')
    recall ← calculateRecall(true_labels, predicted_labels, average='weighted')
    f1_score ← calculateF1Score(true_labels, predicted_labels, average='weighted')
    
    // Visualisasi confusion matrix
    conf_matrix ← calculateConfusionMatrix(true_labels, predicted_labels)
    visualizeHeatmap(conf_matrix)
    
    // Menampilkan classification report
    class_report ← generateClassificationReport(true_labels, predicted_labels)
    
    // Visualisasi hasil training
    visualizeTrainingHistory(history)
    
    RETURN accuracy, precision, recall, f1_score, conf_matrix, class_report
END FUNCTION
```

## 5. Proses Utama
```
PROCEDURE Main():
    // Load dan preprocess data
    X_train, X_test, y_train, y_test, num_classes ← LoadAndPreprocess()
    
    // Reshape data untuk input CNN
    input_shape ← (X_train.shape[1], 1)
    X_train ← reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test ← reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    // Konstruksi model
    model ← BuildModel(input_shape, num_classes)
    
    // Pelatihan model
    model, history ← TrainModel(model, X_train, y_train)
    
    // Evaluasi model
    metrics ← EvaluateModel(model, X_test, y_test)
    
    // Tampilkan hasil
    printResults(metrics)
END PROCEDURE
```
