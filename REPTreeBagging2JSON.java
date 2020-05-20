import weka.classifiers.*;
import weka.classifiers.trees.*;
import weka.classifiers.meta.Bagging;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import java.util.ArrayList;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.PrintWriter;

import java.lang.reflect.*;

public class REPTreeBagging2JSON {
	public static void main(String[] args) throws Exception {
		String filepath = args[0] + ".model";
		Bagging obj = (Bagging) SerializationHelper.read(filepath);
		// obj.getTree(0).setNumDecimalPlaces(17);
		PrintWriter pw = new PrintWriter(new FileWriter(args[0] + ".json"));
		pw.print("[\n");
		
		Class<?> cls = obj.getClass().getSuperclass().getSuperclass().getSuperclass();
		Field field = cls.getDeclaredField("m_Classifiers");
		field.setAccessible(true);
		Classifier[] rt = (Classifier[]) field.get(obj);
                for (int k = 0; k < rt.length; k++) {
		cls = rt[k].getClass();
		field = cls.getDeclaredField("m_Tree");
		field.setAccessible(true);
		Class<?> Tree = Class.forName("weka.classifiers.trees.REPTree$Tree");
		Object n = Tree.cast(field.get(rt[k]));
		cls = n.getClass();
		Field fFeature = cls.getDeclaredField("m_Attribute");
		fFeature.setAccessible(true);
		Field fThreshold = cls.getDeclaredField("m_SplitPoint");
		fThreshold.setAccessible(true);
		Field fChildren = cls.getDeclaredField("m_Successors");
		fChildren.setAccessible(true);
		Field fClassDistribution = cls.getDeclaredField("m_Distribution");
		fClassDistribution.setAccessible(true);
		ArrayList<Object> q = new ArrayList<>();
		ArrayList<Integer> attribute = new ArrayList<>();
		ArrayList<Double> splitPoint = new ArrayList<>();
		ArrayList<ArrayList<Integer>> childrenID = new ArrayList<ArrayList<Integer>>();
		int node_count = 0;
		int node_dequeue = 0;
		q.add(n);
		node_count++;
		
        while (node_dequeue < node_count) {
			n = q.get(node_dequeue);
			if ((int) fFeature.get(n) >= 0) {
				attribute.add((int) fFeature.get(n));
				splitPoint.add((double) fThreshold.get(n));
                Object[] children = (Object[]) fChildren.get(n);
                ArrayList<Integer> lstChildren = new ArrayList<>();
                for (int i = 0; i < children.length; i++) {
				    q.add(children[i]);
                    lstChildren.add(node_count + i);
                }
				childrenID.add(lstChildren);
				node_count += children.length;
			} else {
				attribute.add(-1);
				splitPoint.add(-1.0);
				childrenID.add(new ArrayList<Integer>());
			}
			node_dequeue++;
		}
        // Write JSON
		double[] cnt0 = getCount(fFeature, fClassDistribution, q, childrenID, 0);
		double[] cnt1 = getCount(fFeature, fClassDistribution, q, childrenID, 1);
		pw.print("{\"children\": [");
		for (int i = 0; i < q.size()-1; i++) {
            pw.print("[");
			ArrayList<Integer> lstChildren = childrenID.get(i);
            for (int j = 0; j < lstChildren.size()-1; j++) {
                pw.print(lstChildren.get(j) + ",");
            }
            pw.print(lstChildren.get(lstChildren.size()-1) + "],");
		}
        pw.print("[");
        ArrayList<Integer> lstChildren = childrenID.get(q.size()-1);
        for (int j = 0; j < lstChildren.size()-1; j++) {
            pw.print(lstChildren.get(j) + ",");
        }
        pw.print(lstChildren.get(lstChildren.size()-1) + "]],\n");

		pw.print("\"feature\": [");
		for (int i = 0; i < q.size()-1; i++) {
			pw.print(attribute.get(i) + ",");
		}
		pw.print(attribute.get(q.size()-1) + "],\n");
		pw.print("\"threshold\": [");
		for (int i = 0; i < q.size()-1; i++) {
			pw.print(splitPoint.get(i) + ",");
		}
		pw.print(splitPoint.get(q.size()-1) + "],\n");
		pw.print("\"value\": [");
		for (int i = 0; i < q.size()-1; i++) {
			pw.print(cnt1[i] / (cnt0[i] + cnt1[i]) + ",");
		}
		pw.print(cnt1[q.size()-1] / (cnt0[q.size()-1] + cnt1[q.size()-1]) + "],\n");
		pw.print("\"node_sample_weight\": [");
		for (int i = 0; i < q.size()-1; i++) {
			pw.print((cnt0[i] + cnt1[i]) + ",");
		}
		pw.print((cnt0[q.size()-1] + cnt1[q.size()-1]) + "]}" + (k < rt.length-1 ? ",\n" : "\n"));
		}
		pw.print("]\n");
		pw.close();
	}
	private static double[] getCount(Field fFeature, Field fClassDistribution, ArrayList<Object> q, ArrayList<ArrayList<Integer>> childrenID, int c) throws IllegalArgumentException, IllegalAccessException {
		int sz = q.size();
		double[] ret = new double[sz];
		int i = sz - 1;
		while (i >= 0) {
			Object t = q.get(i);
			if ((int) fFeature.get(t) >= 0) {
				ret[i] = 0;
                ArrayList<Integer> lstChildren = childrenID.get(i);
                for (int j = 0; j < lstChildren.size(); j++) {
                    ret[i] += ret[lstChildren.get(j)];
                }
			} else {
				ret[i] = ((double[]) fClassDistribution.get(t))[c];
			}
			i--;
		}
		return ret;
	}
}
