/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.jml.nn;

import java.io.Serializable;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Logger;
import java.util.stream.Collectors;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import com.jml.main.Graph;

/**
 *
 * @author gt902715 <jean-michel.lottier@alten.com>
 */
public class NeuralNetwork implements Serializable {

    public final int id;

    private static final DecimalFormat df = new DecimalFormat("0.00000");
    private static final Logger LOGGER = Logger.getLogger(NeuralNetwork.class.getName());

    private List<Node> inputLayer;
    private List<Node> hiddenLayer;
    private List<Node> contextLayer;
    private List<Node> outputLayer;
    /**
     * <p>
     * Cross Validation error. Edited by
     * {@link #kFoldCrossValidation(int, java.util.List)} method.</p>
     */
    private float cvError;
    /**
     * <p>
     * Counter incremeted after each call of
     * {@link #kFoldCrossValidation(int, java.util.List)} method.
     * <br>The value equals 0 when the method checks the cross validation error
     * rate.</p>
     */
    private int cpt;
    private NeuralNetwork nnCopy;

    private XYSeries series;
    private int cvTurns;

    public static enum Layers {

        INPUT_LAYER,
        HIDDEN_LAYER,
        CONTEXT_LAYER,
        OUTPUT_LAYER
    };

    public NeuralNetwork() {
        this.id = this.hashCode();
        this.cvError = 10000f;
        this.cpt = 0;
    }

    public NeuralNetwork(NeuralNetwork nn) {
        this.id = nn.id;
        this.inputLayer = nn.getInputLayer().stream().map(n -> new Node(n)).collect(Collectors.toList());
        this.hiddenLayer = nn.getHiddenLayer().stream().map(n -> new Node(n)).collect(Collectors.toList());
        this.outputLayer = nn.getOutputLayer().stream().map(n -> new Node(n)).collect(Collectors.toList());
        if (nn.getContextLayer() != null) {
            this.contextLayer = nn.getContextLayer().stream().map(n -> new Node(n)).collect(Collectors.toList());
        }
        this.cvError = nn.getCvError();
    }

    /**
     *
     * @param nbInputNodes
     * @param nbHiddenNodes
     * @param nbOutputNodes
     * @param isRNN true if we want a recurrent neural network otherwise false
     * (need a hidden layer).
     * @param learningRate
     * @throws Exception
     */
    public void init(int nbInputNodes, int nbHiddenNodes, int nbOutputNodes, boolean isRNN, float learningRate) throws Exception {
        if (nbInputNodes <= 0 || nbOutputNodes <= 0) {
            throw new Exception("Number of input and output node must be positive.");
        }

        inputLayer = new ArrayList<>();
        hiddenLayer = new ArrayList<>();
        outputLayer = new ArrayList<>();

        for (int i = 1; i <= nbInputNodes; ++i) {
            Node inputNode = new Node(i, learningRate);
            if (nbHiddenNodes > 0) {
                for (int j = 1; j <= nbHiddenNodes; ++j) {
                    Node hiddenNode = findNodeByID(nbInputNodes + j);
                    if (hiddenNode == null) {
                        hiddenNode = new Node(nbInputNodes + j, learningRate);
                        if (isRNN) {
                            Node contextNode = new Node(nbInputNodes + nbHiddenNodes + nbOutputNodes + j, learningRate);
                            hiddenNode.connectNode(contextNode.id);
                            hiddenNode.initWeight(contextNode.id);
                            contextNode.connectNode(hiddenNode.id);
                            contextNode.setOutput(0f);
                            if (contextLayer == null) {
                                contextLayer = new ArrayList<>();
                            }
                            contextLayer.add(contextNode);
                        }
                        for (int k = 1; k <= nbOutputNodes; ++k) {
                            Node outputNode = findNodeByID(nbInputNodes + nbHiddenNodes + k);
                            if (outputNode == null) {
                                outputNode = new Node(nbInputNodes + nbHiddenNodes + k, learningRate);
                                outputLayer.add(outputNode);
                            }
                            hiddenNode.connectNode(outputNode.id);
                            outputNode.initWeight(hiddenNode.id);
                        }
                        hiddenLayer.add(hiddenNode);
                    }
                    inputNode.connectNode(hiddenNode.id);
                    hiddenNode.initWeight(inputNode.id);
                }
            } else {
                for (int k = 1; k <= nbOutputNodes; ++k) {
                    Node outputNode = findNodeByID(nbInputNodes + k);
                    if (outputNode == null) {
                        outputNode = new Node(nbInputNodes + k, learningRate);
                        outputLayer.add(outputNode);
                    }
                    inputNode.connectNode(outputNode.id);
                    outputNode.initWeight(inputNode.id);
                }
            }
            inputLayer.add(inputNode);
        }

        if (isRNN) {
            hiddenLayer.stream().forEach((hiddenNode) -> {
                for (Node contextNode : contextLayer) {
                    if (!hiddenNode.getInputWeights().containsKey(contextNode.id)) {
                        hiddenNode.initWeight(contextNode.id);
                        contextNode.connectNode(hiddenNode.id);
                    }
                }
            });
        }
    }

    /**
     *
     * @param nodeID
     * @return
     */
    public Node findNodeByID(int nodeID) {
        Node node;

        node = findNodeByID(nodeID, Layers.INPUT_LAYER);
        if (node != null) {
            return node;
        }

        node = findNodeByID(nodeID, Layers.HIDDEN_LAYER);
        if (node != null) {
            return node;
        }

        node = findNodeByID(nodeID, Layers.CONTEXT_LAYER);
        if (node != null) {
            return node;
        }

        node = findNodeByID(nodeID, Layers.OUTPUT_LAYER);

        return node;
    }

    /**
     *
     * @param nodeID
     * @param layer
     * @return
     */
    public Node findNodeByID(final int nodeID, Layers layer) {
        Node n = null;
        switch (layer) {
            case INPUT_LAYER:
                n = inputLayer.stream().filter(node -> node.id == nodeID).findFirst().orElse(null);
                break;
            case HIDDEN_LAYER:
                if (hiddenLayer != null && hiddenLayer.size() > 0) {
                    n = hiddenLayer.stream().filter(node -> node.id == nodeID).findFirst().orElse(null);
                }
                break;
            case CONTEXT_LAYER:
                if (contextLayer != null && contextLayer.size() > 0) {
                    n = contextLayer.stream().filter(node -> node.id == nodeID).findFirst().orElse(null);
                }
                break;
            case OUTPUT_LAYER:
                n = outputLayer.stream().filter(node -> node.id == nodeID).findFirst().orElse(null);
                break;
        }

        return n;
    }

    /**
     *
     * @param sample
     * @throws Exception
     */
    public void feedForward(Sample sample) throws Exception {
//		System.out.println("++++++++++++++++ START feedForward ++++++++++++++++++");
        if (sample == null || sample.getValues() == null || sample.getValues().isEmpty() || sample.getValues().size() != inputLayer.size()) {
            throw new Exception("The list of inputs must have the same size as inputLayer: " + inputLayer.size());
        }

        hiddenLayer.stream().forEach(n -> {
            n.setSigma(0f);
        });
        outputLayer.stream().forEach(n -> {
            n.setSigma(0f);
        });

        List<Float> values = new ArrayList<>(sample.getValues());
        for (Node inputNode : inputLayer) {
            inputNode.setOutput(values.get(0));
//			System.out.println("NODE " + inputNode.id + " --> OUTPUT " + inputNode.getOutput());
            values.remove(0);
            if (hiddenLayer != null && !hiddenLayer.isEmpty()) {
                hiddenLayer.stream().filter(n -> n.getInputWeights().containsKey(inputNode.id)).forEach(n -> {
                    n.calculateSigma(inputNode);
                });
            } else {
                outputLayer.stream().filter(n -> n.getInputWeights().containsKey(inputNode.id)).forEach(n -> {
                    n.calculateSigma(inputNode);
                });
            }
        }

        if (contextLayer != null && !contextLayer.isEmpty()) {
            contextLayer.stream().forEach((contextNode) -> {
                hiddenLayer.stream().filter(n -> n.getInputWeights().containsKey(contextNode.id)).forEach(n -> {
                    n.calculateSigma(contextNode);
                });
            });
        }

        for (Node hiddenNode : hiddenLayer) {
            hiddenNode.calculateSigma(null); //sigma calculation with bias
            hiddenNode.calculateOutput();
            if (contextLayer != null && !contextLayer.isEmpty()) {
                contextLayer.stream().filter(n -> hiddenNode.getOutputNodeId().contains(n.id)).findFirst().ifPresent(n -> {
                    n.setOutput(hiddenNode.getOutput());
                });
            }
//			System.out.println("NODE " + hiddenNode.id + " --> WEIGHTS " + hiddenNode.getInputWeights() + " --> SIGMA " + hiddenNode.getSigma() + " --> OUTPUT " + hiddenNode.getOutput());
            outputLayer.stream().filter(n -> n.getInputWeights().containsKey(hiddenNode.id)).forEach(n -> {
                n.calculateSigma(hiddenNode);
            });
        }

        for (Node outputNode : outputLayer) {
            outputNode.calculateSigma(null); //sigma calculation with bias
            outputNode.calculateOutput();
//			System.out.println("NODE " + outputNode.id + " --> WEIGHTS " + outputNode.getInputWeights() + " --> SIGMA " + outputNode.getSigma() + " --> OUTPUT " + outputNode.getOutput());
        }
//		System.out.println("++++++++++++++++ END feedForward ++++++++++++++++++");
    }

    /**
     *
     * @param examples
     * @throws Exception
     */
    public void backwardPropagation(List<Sample> examples) throws Exception {
//		System.out.println("++++++++++++++++ START backwardPropagation +++++++++++++++++++=");
        if (examples == null || examples.isEmpty()) {
            throw new Exception("Input parameters cannot be null or empty.");
        }
        List<Sample> _examples = examples.stream().map(sample -> new Sample<>(sample)).collect(Collectors.toList());

        for (int i = 0; i < _examples.size(); ++i) {
            if (!_examples.get(i).isExample()) {
                throw new Exception("All samples must be examples, hence they must contain a list of desired outputs to ajust the neural network.");
            }
            this.feedForward(_examples.get(i));

            for (Node inputNode : inputLayer) {
                float iSumWeightsAndDelta = 0;
                for (int hiddenNodeID : inputNode.getOutputNodeId()) {
                    Node hiddenNode = findNodeByID(hiddenNodeID, Layers.HIDDEN_LAYER);
                    if (hiddenNode != null) {
                        if (!hiddenNode.isDeltaCalculated()) {
                            float hSumWeightsAndDelta = 0;
                            for (int outputNodeID : hiddenNode.getOutputNodeId()) {
                                Node outputNode = findNodeByID(outputNodeID, Layers.OUTPUT_LAYER);
                                if (!outputNode.isDeltaCalculated()) {
//								System.out.println("y : " + Float.valueOf(String.valueOf(_examples.get(i).getDesiredOutputs().get(outputNode.id - hiddenLayer.size() - inputLayer.size() - 1))));
                                    outputNode.calculateDelta(Float.valueOf(String.valueOf(_examples.get(i).getDesiredOutputs().get(outputNode.id - hiddenLayer.size() - inputLayer.size() - 1))), true);
//								System.out.println("Node: " + outputNode.id + ", output: " + outputNode.getOutput() + ", desiredOutput: " + _examples.get(i).getDesiredOutputs().get(outputNode.id - hiddenLayer.size() - inputLayer.size() - 1) + ", delta: " + outputNode.getDelta() + ", sigma: " + outputNode.getSigma());
                                }
                                hSumWeightsAndDelta += outputNode.getInputWeights().get(hiddenNode.id) * outputNode.getDelta();
                                outputNode.reassessWeight(hiddenNode.id, hiddenNode.getOutput());
                            }
                            hiddenNode.calculateDelta(hSumWeightsAndDelta, false);
                        }
                        iSumWeightsAndDelta += hiddenNode.getInputWeights().get(inputNode.id) * hiddenNode.getDelta();
                        hiddenNode.reassessWeight(inputNode.id, inputNode.getOutput());
                        if (contextLayer != null && !contextLayer.isEmpty()) {
                            contextLayer.stream().forEach(n -> {
                                hiddenNode.reassessWeight(n.id, n.getOutput());
                            });
                        }
                    } else {
                        Node outputNode = findNodeByID(hiddenNodeID, Layers.OUTPUT_LAYER);
                        if (!outputNode.isDeltaCalculated()) {
                            outputNode.calculateDelta(Float.valueOf(String.valueOf(_examples.get(i).getDesiredOutputs().get(outputNode.id - inputLayer.size() - 1))), true);
                        }
                        iSumWeightsAndDelta += outputNode.getInputWeights().get(inputNode.id) * outputNode.getDelta();
                        outputNode.reassessWeight(inputNode.id, inputNode.getOutput());
                    }
                }
                inputNode.calculateDelta(iSumWeightsAndDelta, false);
            }
        }
//		System.out.println("++++++++++++++++ END backwardPropagation +++++++++++++++++++=");
    }

    /**
     *
     * @param k
     * @param examples
     * @param drawLineChart
     * @throws Exception
     */
    public void kFoldCrossValidation(int k, List<Sample> examples, boolean drawLineChart) throws Exception {
//		System.out.println("++++++++++++++++++ START k fold cross validation " + this.id + " +++++++++++++++++++++++++");
        if (k < 0) {
            throw new Exception("paramter k must be positive.");
        }

        if (examples == null || examples.isEmpty()) {
            throw new Exception("examples and desiredOutputs parameters cannot be null or empty.");
        }

        if (this.cpt >= 5 || this.nnCopy == null) {
            this.nnCopy = new NeuralNetwork(this);
            this.cpt = 0;
        }

        int nbTurn = k == 0 ? 1 : (Integer) examples.size() / k;
        int fromIndex = 0;
        float _cvError = 0f;
        for (int i = 1; i <= nbTurn; ++i) {
//			System.out.println("TURN: " + i);
            List<Sample> _examples = examples.stream().map(sample -> new Sample<>(sample)).collect(Collectors.toList());
            List<Sample> testingSamples = new ArrayList<>(_examples.subList(fromIndex, (i == nbTurn ? _examples.size() - 1 : fromIndex + k)));
            List<Sample> trainingSamples = new ArrayList<>(_examples.subList(0, fromIndex));
            if (fromIndex + k + 1 < _examples.size()) {
                trainingSamples.addAll(_examples.subList(fromIndex + k + 1, _examples.size() - 1));
            }
            fromIndex += k;

            this.backwardPropagation(trainingSamples);

            float mse = 0; // mean squared error
            for (Sample testSample : testingSamples) {
                this.feedForward(testSample);
//				System.out.println("desiredOutput: " + testSample.getDesiredOutputs().get(0) + ", output: " + this.getOutputLayer().get(0).getOutput());
                float e = this.outputLayer.stream().map((node) -> (float) Math.pow(Math.abs(node.getOutput() - Float.valueOf(String.valueOf(testSample.getDesiredOutputs().get(node.id - inputLayer.size() - hiddenLayer.size() - 1)))), 2))
                        .reduce(0f, (accumulator, _item) -> accumulator + _item);
//				System.out.println("e = " + e);
                mse += e / this.outputLayer.size();
            }
            mse /= testingSamples.size();
//			System.out.println("mse: " + mse);
            _cvError += mse;
        }
        this.cpt++;
        this.cvError = _cvError / k;
//		System.out.println("previous cvError: " + nn.getCvError() + " current cvError: " + this.cvError);
        if (series == null) {
            cvTurns = 0;
            series = new XYSeries(this.id);
        }
        series.add(cvTurns, this.cvError);
        cvTurns++;

        if (this.cpt < 5 || (this.cpt >= 5 && Float.valueOf(df.format(this.cvError).replace(",", ".")) < Float.valueOf(df.format(this.nnCopy.getCvError()).replace(",", ".")))) {
            kFoldCrossValidation(k, examples, drawLineChart);
        } else {
            this.cpt = 0;
            this.cvError = this.nnCopy.getCvError();
            this.inputLayer = this.nnCopy.getInputLayer();
            this.hiddenLayer = this.nnCopy.getHiddenLayer();
            this.outputLayer = this.nnCopy.getOutputLayer();
            if (this.nnCopy.getContextLayer() != null) {
                this.contextLayer = this.nnCopy.getContextLayer();
            }

            if (drawLineChart) {
                Graph graph = new Graph();
                graph.drawLineChart("error rate for neural network " + this.id, new XYSeriesCollection(series));
                series = null;
            }
        }
    }

    public List<Node> getInputLayer() {
        return inputLayer;
    }

    public List<Node> getHiddenLayer() {
        return hiddenLayer;
    }

    public List<Node> getContextLayer() {
        return contextLayer;
    }

    public List<Node> getOutputLayer() {
        return outputLayer;
    }

    public float getCvError() {
        return cvError;
    }

    public void setCvError(float cvError) {
        this.cvError = cvError;
    }
}
