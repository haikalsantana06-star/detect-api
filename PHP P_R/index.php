<?php
// --- 1. KONEKSI DATABASE ---
$host = "localhost";
$user = "iovtmyid_dbupitasmon";     // ganti sesuai user db anda
$pass = "cyMSPkXrkrC7edUJtsCe";         // ganti sesuai password db anda
$db   = "iovtmyid_dbupitasmon"; 

$conn = new mysqli($host, $user, $pass, $db);
if ($conn->connect_error) {
    die("Koneksi gagal: " . $conn->connect_error);
}

// --- 2. QUERY SEMUA DATA DARI TABEL deteksi_objek ---
$sql = "SELECT `id`, `filename`, `hasil_deteksi`, `akurasi`, `waktu_deteksi`, `ruangan_id` 
        FROM `deteksi_objek` 
        ORDER BY `id` DESC";

$result = $conn->query($sql);
?>
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>AI Based Visual Sensing</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.0-beta1/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.0-beta1/dist/js/bootstrap.bundle.min.js"></script>
    
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest/dist/tf.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@teachablemachine/image@latest/dist/teachablemachine-image.min.js"></script>

    <style>
        .status-badge { font-size: 0.85rem; padding: 4px 8px; border-radius: 4px; display: inline-block; margin-top: 5px; min-width: 120px; text-align: center; }
        .is-person { background-color: #d1e7dd; color: #0f5132; border: 1px solid #badbcc; }
        .no-person { background-color: #f8d7da; color: #842029; border: 1px solid #f5c2c7; }
        .loading-ai { background-color: #fff3cd; color: #664d03; }
    </style>
</head>
<body>
<div class="container" style="padding-top:30px;">
    <div class="d-flex justify-content-center"><h1>AI Based Visual Sensing</h1></div>
    <hr class="mt-2 mb-5">

    <div class="row text-center text-lg-start">
        <?php
        $count = 0;
        if ($result && $result->num_rows > 0) {
            while ($row = $result->fetch_assoc()) {
                $count++;
                // Menentukan path gambar berdasarkan ruangan_id
                $filePath = 'ruang_' . $row['ruangan_id'] . '/' . $row['filename'];
                ?>
                <div class="col-lg-3 col-md-4 col-6 mb-4">
                    <div class="card h-100 shadow-sm">
                        <img id="img-<?php echo $count; ?>" class="card-img-top img-thumbnail gallery-img" 
                             src="<?php echo htmlspecialchars($filePath); ?>" 
                             alt="ESP32 Image" 
                             crossorigin="anonymous"
                             data-id="<?php echo $row['id']; ?>"
                             data-filename="<?php echo htmlspecialchars($row['filename']); ?>"
                             data-ruangan="<?php echo $row['ruangan_id']; ?>"
                             onerror="this.onerror=null; this.src='https://via.placeholder.com/300x200?text=Gambar+Tidak+Ada';">
                        
                        <div class="card-body text-center">
                            <p class="small mb-1 text-truncate" title="<?php echo htmlspecialchars($row['filename']); ?>">
                                <strong><?php echo htmlspecialchars($row['filename']); ?></strong>
                            </p>
                            
                            <!-- Badge Status AI Loading / Hasil -->
                            <div id="label-<?php echo $count; ?>" class="status-badge loading-ai">
                                <span class="spinner-border spinner-border-sm"></span> Scanning...
                            </div>

                            <p class="text-muted mt-2 mb-0" style="font-size: 0.75rem;">
                                🕒 <?php echo htmlspecialchars($row['waktu_deteksi']); ?><br>
                                📍 Ruangan: <?php echo htmlspecialchars($row['ruangan_id']); ?>
                            </p>
                        </div>
                    </div>
                </div>
                <?php
            }
        } else {
            echo "<p class='text-center w-100'>Tidak ada data deteksi ditemukan di database.</p>";
        }
        $conn->close();
        ?>
    </div>
</div>

<script type="text/javascript">
let currentFileCount = null;

function checkUpdates() {
    fetch('check_new.php')
        .then(response => response.text())
        .then(count => {
            if (currentFileCount === null) {
                currentFileCount = count;
            } else if (count !== currentFileCount) {
                console.log("Ada data baru! Mengupdate halaman...");
                location.reload();
            }
        })
        .catch(err => console.error("Gagal cek update:", err));
}

// Cek update setiap 5 detik
setInterval(checkUpdates, 5000);

const MODEL_URL = "./my_model/"; 
let model, maxPredictions;

async function initAI() {
    console.log("Memulai loading model...");
    const modelURL = MODEL_URL + "model.json";
    const metadataURL = MODEL_URL + "metadata.json";

    try {
        model = await tmImage.load(modelURL, metadataURL);
        maxPredictions = model.getTotalClasses();
        console.log("Model berhasil dimuat. Jumlah Class:", maxPredictions);
        
        // Jalankan scan untuk semua elemen gambar hasil query database
        scanGallery();
    } catch (e) {
        console.error("Gagal load model! Pastikan folder 'my_model' berisi model.json, metadata.json, dan weights.bin", e);
        document.querySelectorAll('.status-badge').forEach(el => el.innerHTML = "❌ Model Error");
    }
}

async function scanGallery() {
    const images = document.querySelectorAll('.gallery-img');
    console.log("Menscan " + images.length + " gambar dari database...");
    
    for (let i = 0; i < images.length; i++) {
        const imgElement = images[i];
        const labelDiv = document.getElementById(`label-${i+1}`);
        const fileName = imgElement.getAttribute('data-filename');
        const dbId = imgElement.getAttribute('data-id');

        if (!imgElement.complete) {
            await new Promise(resolve => { imgElement.onload = resolve; });
        }

        try {
            const prediction = await model.predict(imgElement);
            let bestClass = "";
            let bestProb = 0;

            for (let j = 0; j < maxPredictions; j++) {
                if (prediction[j].probability > bestProb) {
                    bestProb = prediction[j].probability;
                    bestClass = prediction[j].className;
                }
            }

            const namaClassOrang = "Ada: Hari"; 
            const threshold = 0.60;
            let statusFinal = "";

            if (bestClass === namaClassOrang && bestProb >= threshold) {
                statusFinal = "Ada Orang";
                labelDiv.innerHTML = `✅ ${statusFinal} (${(bestProb * 100).toFixed(0)}%)`;
                labelDiv.className = "status-badge is-person";
            } else {
                statusFinal = "Tidak Ada Orang";
                labelDiv.innerHTML = `❌ ${statusFinal}`;
                labelDiv.className = "status-badge no-person";
            }

            // Kirim hasil pemindaian AI terbaru ke database
            saveToDatabase(dbId, fileName, statusFinal, (bestProb * 100).toFixed(2));

        } catch (err) {
            console.error("Gagal memprediksi ID " + dbId + ":", err);
            labelDiv.innerHTML = "⚠️ Error AI";
        }
    }
}

// Fungsi untuk update/kirim hasil deteksi ke PHP via AJAX
function saveToDatabase(id, filename, hasil, akurasi) {
    if (sessionStorage.getItem('sent_' + id)) {
        console.log("Skip kirim ke DB: ID " + id + " sudah diproses di sesi ini.");
        return; 
    }

    const formData = new FormData();
    formData.append('id', id);
    formData.append('filename', filename);
    formData.append('hasil', hasil);
    formData.append('akurasi', akurasi);

    fetch('save_detection.php', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        console.log("Database Sync ID " + id + ":", data.status);
        if(data.status === 'inserted' || data.status === 'updated' || data.status === 'success') {
            sessionStorage.setItem('sent_' + id, 'true');
        }
    })
    .catch(error => {
        console.error("Gagal simpan ke database:", error);
    });
}

window.onload = initAI;
</script>

</body>
</html>