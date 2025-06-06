library(Rmisc);
library(ggplot2);
library(data.table);

# initialize the data frame
df = data.frame(col1 = numeric(), col2 = character(), col2 = numeric(), stringsAsFactors = FALSE);

df[1, ] <- c(10, "cpu", 0.02);
df[2, ] <- c(100, "cpu", 0.01);
df[3, ] <- c(1000, "cpu", 0.01);
df[4, ] <- c(10000, "cpu", 0.01);
df[5, ] <- c(100000, "cpu", 0.01);
df[6, ] <- c(1000000, "cpu", 0.05);
df[7, ] <- c(10000000, "cpu", 0.06);
df[8, ] <- c(100000000, "cpu", 0.95);

df[9, ] <- c(10, "gpu", 48.08);
df[10, ] <- c(100, "gpu", 46.02);
df[11, ] <- c(1000, "gpu", 50.42);
df[12, ] <- c(10000, "gpu", 89.72);
df[13, ] <- c(100000, "gpu", 393.00);
df[14, ] <- c(1000000, "gpu", 2413.66);
df[15, ] <- c(10000000, "gpu", 19405.30);
df[16, ] <- c(100000000, "gpu", 201918.00);

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
    ylab("latency (usec)") +
    geom_point(size=4.4) +
    geom_line(linewidth=0.8) +
    ggtitle("VectorSUM (CPU vs GPU)") +
    theme(legend.position="bottom") +
    theme(legend.title= element_blank(), legend.text=element_text(size=14,face="bold")) +
    theme(axis.text=element_text(size=14), axis.title=element_text(size=16,face="bold"), plot.title=element_text(size=18,face="bold")) +
    scale_x_log10() +
    theme(aspect.ratio=0.5);

# saving the plot in a pdf file
ggsave("vectorSUM.pdf", plot);
