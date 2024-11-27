        # Botón "Generar Reporte" (después de "Elegir Imagen")
        self.boton_reporte = tk.Button(
            text="Generar Reporte",
            bg="#4A90E2",  # Color de fondo del botón
            fg="#ffffff",  # Color del texto
            font=("Verdana", 9, "bold"),
            borderwidth=0,
            command=self.generar_reporte
        )
        self.boton_reporte.place(x=480, y=70, width=120) 