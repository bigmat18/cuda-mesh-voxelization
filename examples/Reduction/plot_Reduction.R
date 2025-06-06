library(Rmisc);
library(ggplot2);
library(data.table);

# initialize the data frame
df = data.frame(col1 = numeric(), col2 = character(), col2 = numeric(), stringsAsFactors = FALSE);

df[1, ] <- c(262144, "naive", 140.087);
df[2, ] <- c(524288, "naive", 149.64);
df[3, ] <- c(1048576, "naive", 188.493);
df[4, ] <- c(2097152, "naive", 244.819);
df[5, ] <- c(4194304, "naive", 361.577);
df[6, ] <- c(8388608, "naive", 524.863);

df[7, ] <- c(262144, "blocked", 139.881);
df[8, ] <- c(524288, "blocked", 141.295);
df[9, ] <- c(1048576, "blocked", 169.149);
df[10, ] <- c(2097152, "blocked", 190.336);
df[11, ] <- c(4194304, "blocked", 255.94);
df[12, ] <- c(8388608, "blocked", 387.015);

df[13, ] <- c(262144, "strided", 134.551);
df[14, ] <- c(524288, "strided", 135.033);
df[15, ] <- c(1048576, "strided", 156.242);
df[16, ] <- c(2097152, "strided", 175.195);
df[17, ] <- c(4194304, "strided", 243.223);
df[18, ] <- c(8388608, "strided", 333.284);

df[19, ] <- c(262144, "smem", 130.143);
df[20, ] <- c(524288, "smem", 127.468);
df[21, ] <- c(1048576, "smem", 153.567);
df[22, ] <- c(2097152, "smem", 170.527);
df[23, ] <- c(4194304, "smem", 234.545);
df[24, ] <- c(8388608, "smem", 310.167);

df[25, ] <- c(262144, "coarsening", 129.744);
df[26, ] <- c(524288, "coarsening", 126.601);
df[27, ] <- c(1048576, "coarsening", 146.013);
df[28, ] <- c(2097152, "coarsening", 168.753);
df[29, ] <- c(4194304, "coarsening", 222.867);
df[30, ] <- c(8388608, "coarsening", 286.526);

# set the names of the columns
colnames(df) <- c("size", "type", "latency");

# convert the columns in numeric data type
df$size <- as.numeric(as.character(df$size));
df$latency <- as.numeric(as.character(df$latency));

print(df)

# plotting
plot <- ggplot(df, aes(x = size, y = latency, color = type, group = type)) +
    theme_bw() +
    theme(plot.title = element_text(hjust = 0.5)) +
    xlab("vector size (elements)") +
    ylab("kernel latency (usec)") +
    geom_point(size=4.4) +
    geom_line(linewidth=0.8) +
    ggtitle("Reduce") +
    theme(legend.position="bottom") +
    theme(legend.title= element_blank(), legend.text=element_text(size=14,face="bold")) +
    theme(axis.text=element_text(size=14), axis.title=element_text(size=16,face="bold"), plot.title=element_text(size=18,face="bold")) +
    theme(aspect.ratio=0.5);

# saving the plot in a pdf file
ggsave("reduce.pdf", plot);
