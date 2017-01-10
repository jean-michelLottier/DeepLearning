/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.jml.nn;

import java.io.Serializable;
import static java.lang.Math.tanh;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 *
 * @author gt902715 <jean-michel.lottier@alten.com>
 */
public class Node implements Serializable {

    /**
     * Node id
     */
    public final int id;

    private static final DecimalFormat df = new DecimalFormat("0.000");

    /**
     * weights
     */
    private Map<Integer, Float> inputWeights;

    private List<Integer> outputNodeId;

    private float weightBias;

    private float output;

    private float sigma;

    private float delta;

    private boolean deltaCalculated;

    private final float learningRate;

    private ActivationFunction activationFunction = ActivationFunction.SIGMOID;

    public Node(int id, float alpha) {
        this.id = id;
        this.learningRate = alpha;
        this.sigma = 0;
        this.deltaCalculated = false;
        Random r = new Random();
        this.weightBias = Float.valueOf(df.format(-0.5f + r.nextFloat()).replace(",", "."));
    }

    public Node(int id, float alpha, ActivationFunction activationFunction) {
        this.id = id;
        this.learningRate = alpha;
        this.sigma = 0;
        this.deltaCalculated = false;
        this.activationFunction = activationFunction;
        Random r = new Random();
        this.weightBias = Float.valueOf(df.format(-0.5f + r.nextFloat()).replace(",", "."));
    }

    public Node(Node n) {
        this.id = n.id;
        this.learningRate = n.getLearningRate();
        this.sigma = n.getSigma();
        this.deltaCalculated = n.isDeltaCalculated();
        this.weightBias = n.getWeightBias();
        this.delta = n.getDelta();
        if (n.getInputWeights() != null) {
            this.inputWeights = new HashMap<>(n.getInputWeights());
        }
        if (n.getOutputNodeId() != null) {
            this.outputNodeId = new ArrayList<>(n.getOutputNodeId());
        }
        this.output = n.getOutput();
        this.activationFunction = n.activationFunction;
    }

    private float sigmoid(float input) {
        return (float) (1 / (1 + Math.exp(-input)));
    }

    private float pieceWiseLinear(float input) {
        if (input >= 0.5f) {
            return 1;
        } else if (input > -0.5f && input < 0.5f) {
            return input + 0.5f;
        } else {
            return 0;
        }
    }

    private float hyperbolicTangent(float input) {
        return ((float) (Math.exp(input) - Math.exp(-input))) / ((float) (Math.exp(input) + Math.exp(-input)));
    }

    public void connectNode(int nodeID) {
        if (outputNodeId == null) {
            outputNodeId = new ArrayList<>();
        }
        outputNodeId.add(nodeID);
    }

    public void initWeight(int nodeID) {
        if (inputWeights == null) {
            inputWeights = new HashMap<>();
        }

        Random r = new Random();
        inputWeights.put(nodeID, Float.valueOf(df.format(-0.5f + r.nextFloat()).replace(",", ".")));
    }

    public void calculateSigma(Node node) {
        this.deltaCalculated = false;
        if (node == null) {
            this.sigma += this.weightBias * 1;
        } else {
            this.sigma += node.output * this.inputWeights.get(node.id);
        }
    }

    public void calculateOutput() {
        switch (this.activationFunction) {
            case HYPERBOLIC_TANGENT:
                this.output = hyperbolicTangent(this.sigma);
                break;
            case PIECE_WISE_LINEAR:
                this.output = pieceWiseLinear(this.sigma);
                break;
            default:
                this.output = sigmoid(this.sigma);
                break;
        }
    }

    public void calculateDelta(float y, boolean isOutputNode) {
        float derivedActivationFunc;
        switch (this.activationFunction) {
            case HYPERBOLIC_TANGENT:
                derivedActivationFunc = 1f - (float) Math.pow(hyperbolicTangent(this.sigma), 2);
                break;
            case PIECE_WISE_LINEAR:
                if (this.sigma >= 0.5f || this.sigma <= 0.5f) {
                    derivedActivationFunc = 0f;
                } else {
                    derivedActivationFunc = 1f;
                }
                break;
            default:
                derivedActivationFunc = sigmoid(this.sigma) * (1 - sigmoid(this.sigma));
                break;
        }
        if (isOutputNode) {
            this.delta = derivedActivationFunc * (y - this.output);
        } else {
            this.delta = derivedActivationFunc * y;
        }
        this.deltaCalculated = true;
    }

    public void reassessWeight(int nodeID, float input) {
        float result = this.inputWeights.get(nodeID) + this.learningRate * this.delta * input;
        this.inputWeights.put(nodeID, result);
        this.weightBias += this.learningRate * 1 * this.delta;
    }

    public Map<Integer, Float> getInputWeights() {
        return inputWeights;
    }

    public void setInputWeights(Map<Integer, Float> inputWeights) {
        this.inputWeights = inputWeights;
    }

    public List<Integer> getOutputNodeId() {
        return outputNodeId;
    }

    public void setOutputNodeId(List<Integer> outputNodeId) {
        this.outputNodeId = outputNodeId;
    }

    public float getOutput() {
        return output;
    }

    public void setOutput(float output) {
        this.output = output;
    }

    public float getWeightBias() {
        return weightBias;
    }

    public void setWeightBias(float weightBias) {
        this.weightBias = weightBias;
    }

    public float getDelta() {
        return delta;
    }

    public void setDelta(float delta) {
        this.delta = delta;
    }

    public boolean isDeltaCalculated() {
        return deltaCalculated;
    }

    public void setDeltaCalculated(boolean deltaCalculated) {
        this.deltaCalculated = deltaCalculated;
    }

    public float getSigma() {
        return sigma;
    }

    public void setSigma(float sigma) {
        this.sigma = sigma;
    }

    public float getLearningRate() {
        return learningRate;
    }

    public enum ActivationFunction {

        SIGMOID,
        PIECE_WISE_LINEAR,
        HYPERBOLIC_TANGENT;
    }
}
