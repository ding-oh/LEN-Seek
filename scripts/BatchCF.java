import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;

/** Runs G-LoSA's AssignChemicalFeatures on many PDB files inside ONE JVM (feature caching for batch mode).
 *  usage: java -cp cf:cfbatch BatchCF <list file: one pdb path per line>
 *  Output per input: <path without .pdb>-cf.pdb, exactly as AssignChemicalFeatures writes it. */
public class BatchCF {
    public static void main(String[] args) throws Exception {
        List<String> files = Files.readAllLines(Paths.get(args[0]));
        long t0 = System.nanoTime();
        int n = 0;
        for (String f : files) {
            f = f.trim();
            if (f.isEmpty()) continue;
            AssignChemicalFeatures.main(new String[]{f});
            n++;
        }
        System.err.println("BatchCF files=" + n + " sec=" + (System.nanoTime() - t0) / 1e9);
    }
}
