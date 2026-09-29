<?php
// save_detection.php
$host = "localhost";
$user = "iovtmyid_dbupitasmon";     // ganti sesuai user db anda
$pass = "cyMSPkXrkrC7edUJtsCe";         // ganti sesuai password db anda
$db   = "iovtmyid_dbupitasmon"; 

$conn = new mysqli($host, $user, $pass, $db);

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $filename = $_POST['filename'];
    $hasil    = $_POST['hasil'];
    $akurasi  = $_POST['akurasi'];

    // Menyiapkan Query: Jika filename sudah ada, database akan mengupdate hasil & akurasinya saja
    $stmt = $conn->prepare("INSERT INTO deteksi_objek (filename, hasil_deteksi, akurasi) 
                            VALUES (?, ?, ?) 
                            ON DUPLICATE KEY UPDATE hasil_deteksi = VALUES(hasil_deteksi), akurasi = VALUES(akurasi)");
    
    $stmt->bind_param("ssd", $filename, $hasil, $akurasi);
    
    if ($stmt->execute()) {
        // Beri tahu JS apakah ini data baru atau update
        if ($conn->affected_rows == 1) {
            echo json_encode(["status" => "inserted", "message" => "Data baru tersimpan"]);
        } else {
            echo json_encode(["status" => "updated", "message" => "Data diperbarui"]);
        }
    } else {
        echo json_encode(["status" => "error", "message" => $conn->error]);
    }
    $stmt->close();
}
$conn->close();
?>