<?php
$dir = 'ruang_1/';
$files = scandir($dir);
$files = array_diff($files, array('.', '..'));
echo count($files); // Mengembalikan jumlah file saat ini
?>