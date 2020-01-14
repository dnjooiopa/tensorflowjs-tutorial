async function run() {
    const trainData = {
        x: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        y: [1.79, 3, 4.75, 9.2, 12.15, 10.75, 16.55, 15.45, 17.4, 20.15, 24.1, 26.85, 27, 26.05, 28.6, 33.9, 32.65, 34.4, 39.45, 39.3, 43.35]
    };

    const dataTrain = {
        x: trainData.x,
        y: trainData.y,
        name: "train data",
        mode: "markers",
        type: "scatter"
    };

    Plotly.newPlot("dataSpace", [dataTrain], {
        width: 700,
        title: "Simple linear regression",
        xaxis: {
            zeroline: true
        }
    });

    const model = tf.sequential();

    model.add(
        tf.layers.dense({
            units: 64,
            inputShape: [1],
            activation: "relu"
        })
    );

    model.add(
        tf.layers.dense({
            units: 1,
            activation: "linear"
        })
    );

    const LEARNING_RATE = 0.005;
    model.compile({
        optimizer: tf.train.adam(LEARNING_RATE),
        loss: "meanSquaredError"
    });

    const trainX = tf.tensor(trainData.x);
    const trainY = tf.tensor(trainData.y);

    await model.fit(trainX, trainY, {
        epochs: 50
    });

    const xpred = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20];
    const ypred = model
        .predict(tf.tensor(xpred))
        .flatten()
        .arraySync();

    const dataPredict = {
        x: xpred,
        y: ypred,
        name: "model line",
        mode: "lines+markers",
        type: "scatter"
    };

    Plotly.newPlot("dataPredict", [dataTrain, dataPredict], {
        width: 700,
        title: "Simple linear regression (after train)",
        xaxis: {
            zeroline: true
        }
    });
}

run();
