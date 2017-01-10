/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.jml.main;

import java.awt.Color;
import java.util.Random;
import javax.swing.JFrame;
import javax.swing.JPanel;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeriesCollection;

/**
 *
 * @author gt902715 <jean-michel.lottier@alten.com>
 */
public class Graph extends JFrame {

    private XYSeriesCollection dataset;
    private JPanel panel;

    public void drawScatterPlot(String title, XYDataset xYDataset) {
        JFreeChart chart = ChartFactory.createScatterPlot(title, "X axis label", "Y axis label", xYDataset);
        XYPlot plot = chart.getXYPlot();
        Random random = new Random();
        for (int i = 0; i < dataset.getSeriesCount(); ++i) {
            plot.getRenderer().setSeriesPaint(i, new Color(random.nextInt(255), random.nextInt(255), random.nextInt(255)));
        }

        panel = new ChartPanel(chart);
        this.add(panel);
        this.pack();
        this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        this.setVisible(true);
    }

    public void drawLineChart(String title, XYDataset xYDataset) {
        JFreeChart chart = ChartFactory.createXYLineChart(title, null, null, xYDataset, PlotOrientation.VERTICAL, true, true, false);
        panel = new ChartPanel(chart);
        this.add(panel);
        this.pack();
        this.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        this.setVisible(true);
    }

    public XYSeriesCollection getDataset() {
        return dataset;
    }

    public void setDataset(XYSeriesCollection dataset) {
        this.dataset = dataset;
    }

    public void cleanDataset() {
        dataset.removeAllSeries();
    }

    public JPanel getPanel() {
        return panel;
    }

    public void setPanel(JPanel panel) {
        this.panel = panel;
    }

}
