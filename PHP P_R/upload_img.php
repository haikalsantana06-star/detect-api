<?php
// based on PHP File Upload basic example https://www.w3schools.com/php/php_file_upload.asp

// 1. KONEKSI KE DATABASE
$host = "localhost";
$user = "iovtmyid_dbupitasmon";     // Sesuaikan dengan user DB Anda
$pass = "cyMSPkXrkrC7edUJtsCe";         // Sesuaikan dengan password DB Anda
$db   = "iovtmyid_dbupitasmon";     // Sesuaikan dengan nama DB Anda

$conn = new mysqli($host, $user, $pass, $db);

if ($conn->connect_error) {
    die("Koneksi database gagal: " . $conn->connect_error);
}

// Set timezone Indonesia/Jakarta
date_default_timezone_set('Asia/Jakarta');

$target_dir = "ruang_1/"; // folder untuk menyimpan gambar

// Buat direktori jika belum ada
if (!file_exists($target_dir)) {
    mkdir($target_dir, 0777, true);
}

// Penamaan file gambar dengan timestamp
$filename_only = date('Y.m.d_H:i:s_') . basename($_FILES["imageFile"]["name"]);
$target_file   = $target_dir . $filename_only;
$uploadOk      = 1;
$imageFileType = strtolower(pathinfo($target_file, PATHINFO_EXTENSION));

// Check if image file is an actual image or fake image
if (isset($_POST["submit"])) {
  $check = getimagesize($_FILES["imageFile"]["tmp_name"]);
  if ($check !== false) {
    // File valid
    $uploadOk = 1;
  } else {
    echo "File is not an image.";
    $uploadOk = 0;
  }
}

// Check if file already exists
if (file_exists($target_file)) {
  echo "Sorry, file already exists.";
  $uploadOk = 0;
}

// Check file size (Maksimal 500KB)
if ($_FILES["imageFile"]["size"] > 500000) {
  echo "Sorry, your file is too large.";
  $uploadOk = 0;
}

// Allow certain file formats
if ($imageFileType != "jpg" && $imageFileType != "png" && $imageFileType != "jpeg" && $imageFileType != "gif") {
  echo "Sorry, only JPG, JPEG, PNG & GIF files are allowed.";
  $uploadOk = 0;
}

// Check if $uploadOk is set to 0 by an error
if ($uploadOk == 0) {
  echo "Sorry, your file was not uploaded.";
// if everything is ok, try to upload file
} else {
  if (move_uploaded_file($_FILES["imageFile"]["tmp_name"], $target_file)) {
    
    // 2. PROSES INSERT KE DATABASE SETELAH GAMBAR BERHASIL DI-UPLOAD
    $hasil_default   = "Belum Terdeteksi"; // Nilai awal sebelum diolah Teachable Machine / AI
    $akurasi_default = 0.0;
    $waktu_deteksi   = date("Y-m-d H:i:s"); // Timestamp saat ini

    $stmt = $conn->prepare("INSERT INTO deteksi_objek (filename, hasil_deteksi, akurasi, waktu_deteksi) VALUES (?, ?, ?, ?)");
    $stmt->bind_param("ssds", $filename_only, $hasil_default, $akurasi_default, $waktu_deteksi);

    if ($stmt->execute()) {
        echo "The file ". basename($_FILES["imageFile"]["name"]). " has been uploaded and saved to database.";
    } else {
        echo "File uploaded, but failed to insert to database: " . $stmt->error;
    }
    
    $stmt->close();

  } else {
    echo "Sorry, there was an error uploading your file.";
  }
}

$conn->close();
?>