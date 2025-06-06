library(Rmisc);
library(ggplot2);
library(data.table);

# initialize the data frame
df = data.frame(col1 = numeric(), col2 = character(), col2 = numeric(), stringsAsFactors = FALSE);

df[1, ] <- c(1, "2M", 4475.92);
df[2, ] <- c(2, "2M", 4055.75);
df[3, ] <- c(4, "2M", 3898.91);
df[4, ] <- c(8, "2M", 3894.53);
df[5, ] <- c(16, "2M", 3938.49);
df[6, ] <- c(32, "2M", 4193.21);
df[7, ] <- c(64, "2M", 4588.73);

# set the names of the columns
colnames(df) <- c("streams", "type", "latency");

# convert the columns in numeric data type
df$streams <- as.numeric(as.character(df$streams));
df$latency <- as.numeric(as.character(df$latency));

print(df)

# plotting
plot <- ggplot(df, aes(x = streams, y = latency, color = type, group = type)) +
    theme_bw() +
    theme(plot.title = element_text(hjust = 0.5)) +
    xlab("number of CUDA streams") +
    ylab("latency (usec)") +
    geom_point(size=4.4) +
    geom_line(linewidth=0.8) +
    ggtitle("VectorSUM (Streamed)") +
    theme(legend.position="bottom") +
    ylim(3000, 5000) +
    scale_x_continuous(breaks = c(1, 2, 4, 8, 16, 32, 64)) +
    theme(legend.title= element_blank(), legend.text=element_text(size=14,face="bold")) +
    theme(axis.text=element_text(size=14), axis.title=element_text(size=16,face="bold"), plot.title=element_text(size=18,face="bold")) +
    theme(aspect.ratio=0.7);

# saving the plot in a pdf file
ggsave("vectorSUM_streamed.pdf", plot);
