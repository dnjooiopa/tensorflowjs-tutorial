const tf = require("@tensorflow/tfjs-node")


async function run(){

    const model = tf.sequential();

    model.add(tf.layers.dense({
        units: 64,
        inputShape: [1],
        activation: "r"
    }))


}